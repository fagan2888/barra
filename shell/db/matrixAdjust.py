
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
import statsmodels.api as sm
from ipdb import set_trace

# exponent weighed adjustment
# f(i,j) = std(i) * std(j) * r(i,j)
# when calculating std and r different half-life are setted
def exponentWeight(frAll, sdate, halfLifeStd = 252, halfLifeR = 84, weight = True):
    
    distance = len(frAll) - len(frAll[frAll.index>=sdate])
    newMatrix = np.zeros((len(frAll[frAll.index>=sdate]),frAll.shape[1],frAll.shape[1]))
    rMatrix = np.zeros(np.shape(newMatrix))
    frStdMatrix = np.zeros((len(frAll[frAll.index>=sdate]),frAll.shape[1],distance+1))
    frRMatrix = np.zeros(np.shape(frStdMatrix))
    stdMatrix = np.zeros((len(frAll[frAll.index>=sdate]),frAll.shape[1],1))
    frMatrix = np.zeros(np.shape(frRMatrix))
    for i in range(len(frAll[frAll.index>=sdate])):
        fr = frAll.iloc[i:i+distance+1]
        frStd = fr.copy()
        frR = fr.copy()
        if weight != True:
            frMatrix[i,:,:] = np.mat(fr.values).T
            continue
        wStd = [ 0.5**((distance - i)/halfLifeStd) for i in range(distance+1)]
        wR = [ 0.5**((distance - i)/halfLifeR) for i in range(distance+1)]
        for column in list(fr.columns):
            frStd.loc[:,column] = fr[column]*wStd
            frR.loc[:,column] = fr[column]*wR
        frStd = np.mat(frStd.values).T
        frR = np.mat(frR.values).T
        std = np.std(frStd, axis = 1) 
        r = np.corrcoef(frR)
        
        #newMatrix[i,:,:] = np.dot(std, std.T) * r
        rMatrix[i,:,:] = r
        stdMatrix[i,:,:] = std
        frStdMatrix[i,:,:] = frStd
        frRMatrix[i,:,:] = frR

    if weight != True:
        return frMatrix

    return stdMatrix, rMatrix, frStdMatrix, frRMatrix

# fr = [factor returns, moments] and sorted by moments
# cov0 = Sigma(fri*fri.T)/t (1~t)
# cov0 can be inplaced by matrix that has already been adjusted by exponentWeight()
# cov = cov0 + Sigma(wi*(covi + covi.T)) (1~q) , wi = 1 - i/(1+q), covi = Sigma(frj*frj+i.T)/t (1~t-i)
# factor return frt  of moment 't' is influenced by moments of 't-1','t-2','t-3' ...... 't-q'
def neweyWestRaw(fr, q):
    
    factorNum = np.shape(fr)[0]
    t = np.shape(fr)[1]
    cov0 = np.zeros((factorNum, factorNum))
    for i in range (t):
        fri = np.mat(fr[:,i].reshape(factorNum,1))
        cov0 = cov0 + np.dot(fri,fri.T)
    cov0 = cov0/t

    sigmaCov = np.zeros((factorNum, factorNum))
    for i in range (1,q+1):
        covi = np.zeros((factorNum, factorNum))
        for j in range (t-i):
            fri = np.mat(fr[:,j].reshape(factorNum,1))
            frj = np.mat(fr[:,j+i].reshape(factorNum,1))
            covi = covi + np.dot(fri, frj.T)
        wi = 1 - i / (1+q)
        sigmaCov = wi * (covi + covi.T) + sigmaCov
    cov = cov0 + sigmaCov / t

    # change cov of daily data into monthly data
    # 21 trade days per month. so monthly return is supposed to be 21 times of daily return
    newMatrix = 21*cov

    return newMatrix

# this newey west adjustment can adjust std matrix and r matrix respectively
def neweyWest(std0, r0, frStd, frR, qStd = 5, qR = 2):

    factorNum = np.shape(r0)[1]
    t = np.shape(r0)[0]
    delta = np.shape(frR)[2]
    r = np.zeros((t,factorNum,factorNum))
    std = np.zeros((t,factorNum,1))
    newMatrix = np.zeros((t,factorNum,factorNum))

    for date in range(t):
        sigmaR = np.zeros((factorNum,factorNum))
        for i in range (1,qR+1):
            fr1 = np.mat(frR[date,:,0:delta-i].reshape(factorNum, delta-i))
            fr2 = np.mat(frR[date,:,i:delta].reshape(factorNum, delta-i))
            ri = np.zeros((factorNum,factorNum))
            for j in range(factorNum):
                for k in range(j,factorNum):
                    ri[j,k] = pearsonr(fr1[j,:].tolist()[0], fr2[k,:].tolist()[0])[0]
                    ri[k,j] = pearsonr(fr1[k,:].tolist()[0], fr2[j,:].tolist()[0])[0]
            wi = 1 - i / (1+qR)
            sigmaR = wi * (ri + ri.T) + sigmaR
        r[date,:,:] = np.mat(r0[date,:,:].reshape((factorNum,factorNum))) + sigmaR / delta
        
        covRaw = neweyWestRaw(frStd[date,:,:].reshape(np.shape(frStd)[1:3]), qStd)
        cov = covRaw.diagonal().T
        stdTmp = np.sqrt(cov)
        std[date,:,:] = stdTmp
        stdTmp = np.mat(stdTmp)

        newMatrix[date,:,:] = np.multiply(np.dot(stdTmp, stdTmp.T), np.mat(r[date,:,:].reshape(factorNum, factorNum)))
    
    return newMatrix

# adjustment based on eigen values is based on neweyWest adjustment
# d0 = u0.T*neweyWestMat*u0  u0 = eigVector
# rm = u0 * bm   fm = cov(rm, rm) dm = um.T*fm*um dmReal = um.T*neweyWest*um
# lamda = sqrt(Sigma(dmReal/dm)/m) (m = 1,2,3 ...... M) 
# gama = alpha*(lamda-1)+1 d0Real = gama**2 * d0 
# eigenMat = u0*d0Real*u0.T
def eigen(neweyWestMatAll, M = 1000, alpha = 1.2, timeWindows = 365):

    def diag(mat, one = False):
        if one == True:
            matNew = np.ones(np.shape(mat))
        else:
            matNew = np.zeros(np.shape(mat))
        for j in range(np.shape(mat)[0]):
            matNew[j,j] = mat[j,j]
        return matNew
 
    t = np.shape(neweyWestMatAll)[0]
    factorNum = np.shape(neweyWestMatAll)[1]
    newMatrix = np.zeros(np.shape(neweyWestMatAll))
    for i in range(t):
        neweyWestMat = np.mat(neweyWestMatAll[i,:,:].reshape(factorNum, factorNum))
        eigValues, eigVectors = np.linalg.eig(neweyWestMat)
        u0 = eigVectors
        d0 = u0.T * neweyWestMat * u0
        #print(np.max(np.cov(u0)/neweyWestMat),np.min(np.cov(u0)/neweyWestMat))
        #set_trace()
        
        # try M times
        sigma = np.zeros((factorNum,factorNum))
        for m in range(M):
            bm = np.zeros((factorNum,timeWindows))
            for k in range(factorNum):
                bm[k,:] = np.random.normal(0, np.sqrt(abs(d0[k,k])),timeWindows)
            #print(bm)
            #set_trace()
            rm = u0 * bm
            fm = np.cov(rm)
            em, um = np.linalg.eig(fm)
            dm = um.T * fm * um
            dmReal = um.T * neweyWestMat * um
            dm = diag(dm, one = True)
            dmReal = diag(dmReal)
            sigma = sigma + dmReal / dm

        sigma = (sigma + abs(sigma)) / 2
        lamda = np.sqrt(sigma/M)
        gama = alpha * (lamda - 1) + 1
        d0Real = np.multiply(np.multiply(gama ,gama), d0)
   
        newMatrix[i,:,:] = u0 * d0Real * u0.T

    return newMatrix

# adjustment based on fluctuation ratio is based on eigen adjustment
# the number of factors is K
# bft = sqrt((frkt/flrkt)**2 /K)  fr is factor return is. flr is fluctuation.
# lamda = sqrt(sigma(bft**2, wt))
# fvar = lamda**2 * feigen
# when it comes to heterogeneity return of stocks, adjustment is based on the output of bayes adjustment
def fluctuation(eigenMatAll, frAll, flrAll, sdate):
    
    t = np.shape(eigenMatAll)[0]
    factorNum = np.shape(eigenMatAll)[1]
    newMatrix = np.zeros(np.shape(eigenMatAll))
    
    frAll = exponentWeight(frAll, sdate, weight = False)
    flrAll = exponentWeight(flrAll, sdate, weight = False)
    delta = np.shape(flrAll)[2]
    
    for date in range(t):
        fr = np.mat(frAll[date,:,:].reshape(factorNum, delta))
        flr = np.mat(flrAll[date,:,:].reshape(factorNum, delta))
        eigenMat = np.mat(eigenMatAll[date,:,:].reshape(factorNum, factorNum))
        sigma = np.zeros((1,delta))        
        for i in range(factorNum): 
            sigma = np.multiply(fr[i,:]/flr[i,:], fr[i,:]/flr[i,:]) + sigma
        sigma[np.isnan(sigma)] = 0
        sigma[np.isinf(sigma)] = 0
        sigma = sigma / factorNum

        w = np.zeros((delta,1))
        for i in range(delta):
            w[i,0] = 0.5**((delta-i-1)/delta)
        w = w / np.sum(w)
            
        lamda = np.dot(sigma, w)[0,0]

        newMatrix[date,:,:] = lamda * eigenMat

    return newMatrix

# structure model adjustment
# sigma = gama * sigmaNeweyWest + (1-gama) * sigmaStruct    sigma is the result,sigmaNeweyWest is std after neweyWest adjustment,
# gama = (min(1, max(0, h/120-0.5))) * (min(1, max(0, exp(1-zu))))    gama is a parameter for judging the quanlity of a resid, the bigger the better  h: h = 252
# sigmaStruct(gama != 1) = E0 * exp(Sigma(Xnk * bk))     Xnk: exposure value  bk: weight of the No.k factor E0: a const
# use ln(sigmaStruct(gama == 1)) = Sigma(Xnk*bk) + en doing linear regression then work out bk and sigmaStruct(gama == 1) == sigmaNeweyWest(gama == 1)
# zu = abs(sigma / sigmaReference -1)    zu is a paramter which has the same function as what gama has the less the better  sigma: covariance of the returns without any adjustment
# sigmaReference = 1/1.35 * (Q3 - Q1)
# Q3 = 3/4*rn Q1 = 1/4 * rn    rn: the return of No.n stock
# all inputs should in form of matrix
def structure(neweyWestAll, rAll, flrAll, exposureAll, dates, h = 252, E0 = 1.05):

    dates.sort()
    n = len(neweyWestAll)
    newList = []
    for date in range(n):
        r = rAll[rAll.index == dates[date]]
        flr = flrAll[flrAll.index == dates[date]]
        exposure = exposureAll[exposureAll.index == dates[date]]
        neweyWestDf = neweyWestAll[date]
        stocks = set(neweyWestDf.columns).intersection(set(r.columns).intersection(set(flr.columns).intersection(set(list(exposure.stock_id)))))
        stocks = list(stocks)
        stocks.sort()
        neweyWestDf = neweyWestDf[stocks].sort_index(axis = 1, ascending = True)
        r = r[stocks].sort_index(axis = 1, ascending = True)
        flr = flr[stocks].sort_index(axis = 1, ascending = True)
        exposure = exposure[exposure.stock_id.isin(stocks)].fillna(0).sort_values(by = 'stock_id', ascending = True)
        r = np.mat(r.T)
        flr = np.mat(flr.T)
        exposure = np.mat(exposure.set_index('stock_id'))
        neweyWestMat = np.mat(neweyWestDf)
        Q3 = 3/4 * r
        Q1 = 1/4 * r
        sigmaReference = 1/1.35 * (Q3 -Q1)
        zu = np.abs(flr / sigmaReference - 1)
        gama = np.zeros(np.shape(zu)) 
        for i in range(len(zu)):
            gama[i,0] = (min(1, max(0, h/120-0.5))) * (min(1, max(0, np.exp(1-zu[i,0]))))
        
        locations = np.where(gama == 1)[0]
        locations2 = np.where(gama < 1)[0]
        if len(locations2) == 0:
            df = pd.DataFrame(neweyWestMat)
            df.columns = stocks
            newList.append(df)
            continue
        sigmaStruct = np.zeros(np.shape(neweyWestMat))
        lnSigma = []
        exposureTmp = np.zeros((len(locations),np.shape(exposure)[1]))
        i = 0
        for location in locations:
            sigmaStruct[location,location] = neweyWestMat[location,location]
            lnSigma.append(neweyWestMat[location,location])
            exposureTmp[i,:] = exposure[location,:]
            i += 1
        lnSigma = np.log(np.mat(lnSigma)).T
        # use generalized linear model
        model = sm.GLM(lnSigma, exposureTmp)
        b = np.mat(model.fit().params).T
        for location in locations2:
            sigmaStruct[location,location] = E0 * np.exp(exposure[location,:] * b)

        matTmp = np.zeros(np.shape(neweyWestMat))
        for i in range(np.shape(neweyWestMat)[0]):
            matTmp[i,i] = gama[i,0]*neweyWestMat[i,i] + (1 - gama[i,0]) * sigmaStruct[i,i]
        df = pd.DataFrame(matTmp)
        df.columns = stocks
        newList.append(df)

    return newList

# bayes adjustment
# sigmaSHn = vn * sigmaSn + (1-vn)*sigman  n = 1,2,3,....,N
# sigmaSn = Sigma(wn*sigman)  n = 1,2,3,......, N
# vn = 1 /(deltaSn/(q*abs(sigman - sigmaSn)) + 1)
# deltaSn = sqrt(Sigma((sigman - sigmaSn)^2)/NSn)
# weight is read from barra_factor_exposure also weight should be normalized beforehand
def bayes(structureAll, weightAll, dates, groups = 10, q = 1):
    
    dates.sort()
    newList = []
    for t in range(len(structureAll)):
        structureDf = structureAll[t]
        weightDf = weightAll[weightAll.index == dates[t]]
        stocks = list(set(structureDf.columns).intersection(set(weightDf['stock_id'])))
        stocks.sort()
        structureDf = structureDf[stocks]
        weight = weightDf[weightDf.stock_id.isin(stocks)].sort_values(by = 'stock_id')['weight']
        structureMat = np.mat(structureDf)

        sigma = np.mat(np.diag(structureMat)).T
        dfw = pd.DataFrame({'sigma':np.diag(structureMat),'weight':weight})
        dfw = dfw.reset_index(drop = True)
        dfw['location'] = list(dfw.index)
        dfw.sort_values(by = 'weight', ascending = True, inplace = True)
        dfw.reset_index(drop = True, inplace = True)
        v = np.zeros((len(dfw),1))
        sigmaS = np.zeros((len(dfw),1))
        for i in range(groups-1):
            df = dfw.iloc[i*(len(dfw)//groups):(i+1)*(len(dfw)//groups)]
            weightSum = sum(df['weight'])
            df.loc[:,'weight'] = df['weight'].apply(lambda x: x/weightSum)
            locations = list(df['location'])
            sigmaTmp = sigma[locations]
            sigmaSTmp = np.ones((len(locations),1))*((df['weight']*df['sigma']).mean())
            deltaS = df['sigma'].std()
            sigmaS[locations] = sigmaSTmp
            v[locations] = 1 /(deltaS/(q*abs(sigmaTmp - sigmaSTmp)) + 1)
        
        df = dfw.iloc[(groups-1)*(len(dfw)//groups):len(dfw)]
        weightSum = sum(df['weight'])
        df.loc[:,'weight'] = df['weight'].apply(lambda x: x/weightSum)
        locations = list(df['location'])
        sigmaTmp = sigma[locations]
        sigmaSTmp = np.ones((len(locations),1))*((df['weight']*df['sigma']).mean())
        deltaS = df['sigma'].std()
        sigmaS[locations] = sigmaSTmp
        v[locations] = 1 /(deltaS/(q*abs(sigmaTmp - sigmaSTmp)) + 1)

        newArray = np.multiply(v, sigmaS) + np.multiply((1-v), sigma)
        newMat = np.zeros(np.shape(structureMat))
        for i in range(len(sigma)):
            newMat[i,i] = newArray[i,0]
        newDf = pd.DataFrame(newMat)
        newDf.columns = stocks
        newList.append(newDf)

    return newList


def exponentWeightOmiga(frAll, sdate, halfLife = 84, weight = True):
    
    distance = len(frAll) - len(frAll[frAll.index>=sdate])
    dfList = []
    for i in range(len(frAll[frAll.index>=sdate])):
        fr = frAll.iloc[i:i+distance+1]
        frTmp = fr.copy()
        frTmp = frTmp.dropna(axis = 1)
        if weight != True:
            dfList.append(frTmp.T)
            continue
        w = [ 0.5**((distance - i)/halfLife) for i in range(distance+1)]
        for column in list(frTmp.columns):
            frTmp.loc[:,column] = fr[column]*w
        
        dfList.append(frTmp.T)

    return dfList


def neweyWestOmiga(resid, q = 5):

    dates = len(resid)
    dfList = []
    for date in range(dates):
        mat = neweyWestRaw(np.mat(resid[date]), q)
        df = pd.DataFrame(mat)
        df.columns = resid[date].T.columns
        dfList.append(df)

    return dfList


def fluctuationOmiga(omigaAll, rAll, flrAll, weightAll, sdate, dates):
  
    weightAll = weightAll.reset_index().set_index(['trade_date','stock_id']).sort_index(axis = 0, ascending = True).unstack()['weight']
    t = len(omigaAll)
    rAll = exponentWeightOmiga(rAll, sdate, weight = False)
    flrAll = exponentWeightOmiga(flrAll, sdate, weight = False)
    weightAll = exponentWeightOmiga(weightAll, sdate, weight = False)
    delta = rAll[0].shape[1]
   
    newList = []
    for date in range(t):
        stocks = list(set(omigaAll[date].columns).intersection(set(rAll[date].index).intersection(set(flrAll[date].index).intersection(set(weightAll[date].index)))))
        stocks.sort()
        r = rAll[date].loc[stocks,:]
        flr = flrAll[date].loc[stocks,:]
        weight = weightAll[date].loc[stocks,:]
        omiga = omigaAll[date]
        omiga.index = omiga.columns
        omiga = omiga.loc[stocks,stocks]
        for k in range (weight.shape[1]):
            weight.iloc[:,k] = weight.iloc[:,k] / sum(weight.iloc[:,k])
        r = np.mat(r)
        flr = np.mat(flr)
        weight = np.mat(weight)
        omiga = np.mat(omiga)
        sigma = np.zeros((1,delta))  
        stockNum = np.shape(omiga)[0]
        for i in range(stockNum): 
            sigma =  np.multiply(weight[i,:], np.multiply(r[i,:]/flr[i,:], r[i,:]/flr[i,:])) + sigma
        sigma[np.isnan(sigma)] = 0
        sigma[np.isinf(sigma)] = 0

        w = np.zeros((delta,1))
        for i in range(delta):
            w[i,0] = 0.5**((delta-i-1)/delta)
        w = w / np.sum(w)

        newMat = np.dot(sigma, w)[0,0] * omiga
        newDf = pd.DataFrame(newMat)
        newDf.columns = stocks
        newList.append(newDf)

    return newList


