
import numpy as np
import pandas as pd
import random
import math
from scipy.stats import pearsonr
import statsmodels.api as sm

# matrix adjustment

def nothing(matrix):
    
    pass
    
    return matrix

# exponent weighed adjustment
# f(i,j) = std(i) * std(j) * r(i,j)
# when calculating std and r different half-life are setted
# fr = [factor returns, moments] and sorted by moments
def exponentWeight(fr, halfLifeStd = 252, halfLifeR = 84):

    w = 0.5**(1 / halfLifeStd) * np.matrix((np.linspace(1,np,shape(fr)[1],np.shape(fr)[1])))
    w = np.diag(w)
    frStd = np.dot(fr, w)
    std = np.std(frStd, axis = 1) 

    w = 0.5**(1 / halfLifeR) * np.matrix((np.linspace(1,np,shape(fr)[1],np.shape(fr)[1])))
    w = np.diag(w)
    frR = np.dot(fr, w)
    r = np.corrcoef(frR)

    newMatrix = np.dot(std, std.T) * r

    return newMatrix

# fr = [factor returns, moments] and sorted by moments
# cov0 = Sigma(fri*fri.T)/t (1~t)
# cov0 can be inplaced by matrix that has already been adjusted by exponentWeight()
# cov = cov0 + Sigma(wi*(covi + covi.T)) (1~q) , wi = 1 - i/(1+q), covi = Sigma(frj*frj+i.T)/t (1~t-i)
# factor return frt  of moment 't' is influenced by moments of 't-1','t-2','t-3' ...... 't-q'
def neweyWestRaw(fr, q, halfLife = None):
    
    cov0 = np.zeros((np.shape(fr)[0],np.shape(fr)[0]))
    t = np.shape(fr)[1]
    for i in range (t):
        cov0 = cov0 + np.dot(fr[:,i],fr[:,i].T)
    cov0 = cov0/t

    # if use exponent weights or not
    if halfLife != None:
        w = 0.5**(1 / halfLife) * np.matrix((np.linspace(1,np,shape(fr)[1],np.shape(fr)[1])))
        w = np.diag(w)
        fr = np.dot(fr, w)
 
    sigmaCov = np.zeros((np.shape(fr)[0],np.shape(fr)[0]))
    t = np.shape(fr)[1]
    for i in range (1,q+1):
        covi = np.zeros((np.shape(fr)[0],np.shape(fr)[0]))
        for j in range (t-i):
            covi = covi + np.dot(fr[:,j], fr[:,i+j].T)
        covi = covi/t
        wi = 1 - i / (1+q)
        sigmaCov = wi * (covi + covi.T) + sigmaCov
    cov = cov0 + cov

    # change cov of daily data into monthly data
    # 22 trade days per month. so monthly return is supposed to be 22 times of daily return
    newMatrix = 22*cov

    return newMatrix

# this newey west adjustment can adjust std matrix and r matrix respectively
def neweyWest(exponentMat, fr, qStd = 5, qR = 2, halfLifeStd = 252, halfLifeR = 84):

    factorNum = np.shape(fr)[0]
    t = np.shape(fr)[1]
    w = 0.5**(1 / halfLifeR) * np.matrix((np.linspace(1,np,shape(fr)[1],np.shape(fr)[1])))
    w = np.diag(w)
    frR = np.dot(fr, w)
    r0 = np.corrcoef(frR)
    sigmaR = np.zeros((factorNum,factorNum))
    for i in range (1,qR+1):
        fr1 = frR[:,0:t-i]
        fr2 = frR[:,i:t]
        ri = np.zeros((factorNum,factorNum))
        for j in range(factorNum):
            for k in range(j,factorNum):
                ri[j,k] = pearsonr(fr1[j,:], fr2[k,:])[0]
                ri[k,j] = pearsonr(fr1[k,:], fr2[j,:])[0]
        wi = 1 - i / (1+qR)
        sigmaR = wi * (ri + ri.T) + sigmaR
    r = r0 + r

    covRaw = neweyWestRaw(fr, qStd, halfLifeStd)
    cov = covRaw.diagonal().T
    std = np.sqrt(cov)

    '''
    w = 0.5**(1 / halfLifeStd) * np.matrix((np.linspace(1,np,shape(fr)[1],np.shape(fr)[1])))
    w = np.diag(w)
    frCov = np.dot(fr, w)
    sigmaCov = np.zeros((factorNum,factorNum))
    for i in range (1,qCov+1):
        fr1 = frCov[:,0:t-i]
        fr2 = frCov[:,i:t]
        covi = np.zeros((factorNum,factorNum))
        for j in range(factorNum):
            for k in range(j,factorNum):
                covi[j,k] = np.std(fr1[j,:])*np.std(fr2[k,:])*r[j,k]
                covi[k,j] = np.std(fr1[k,:])*np.std(fr2[j,:])*r[k,j]
        covi = covi*(t - i)/t
        wi = 1 - i / (1+qCov)
        sigmaCov = wi * (covi + covi.T) + sigmaCov
    
    newMatrix = exponent + sigmaCov
    '''
    newMatrix = np.dot(std * std.T) * r

    return newMatrix

# adjustment based on eigen values is based on neweyWest adjustment
# d0 = u0.T*neweyWestMat*u0  u0 = eigVector
# rm = u0 * bm   fm = cov(rm, rm) dm = um.T*fm*um dmReal = um.T*neweyWest*um
# lamda = sqrt(Sigma(dmReal/dm)/m) (m = 1,2,3 ...... M) 
# gama = alpha*(lamda-1)+1 d0Real = gama**2 * d0 
# eigenMat = u0*d0Real*u0.T
def eigen(neweyWestMat, fr, M = 1000, alpha = 1.2):

    eigValues, eigVectors = np.linalg.eig(neweyWestMat)
    u0 = eigVectors
    d0 = u0.T * neweyWestMat * u0
    
    # try M times
    sigma = np.zeros((M,1))
    for m in range(M):
        bm = np.zeros((np.shape(d0)[0],1))
        for i in range(np.shape(d0)[0]):
            bm[i][0] = random.normalvariate(0, sqrt(d0[i][i]))
        rm = u0 * bm
        fm = np.cov(rm)
        em, um = np.linalg.eig(fm)
        dm = um.T * fm * um
        dmReal = um.T * neweyWest * um
        sigma = sigma + dmReal / dm 

    lamda = np.sqrt(sigma/M)
    gama = alpha * (lamda - 1) + 1
    d0Real = gama**2 * d0
    newMatrix = u0 * d0Real * u0.T

    return newMatrix

# adjustment based on fluctuation ratio is based on eigen adjustment
# the number of factors is K
# bft = sqrt((frkt/flrkt)**2 /K)  fr is factor return is. flr is fluctuation.
# lamda = sqrt(sigma(bft**2, wt))
# fvar = lamda**2 * feigen
# when it comes to heterogeneity return of stocks, adjustment is based on the output of bayes adjustment
def fluctuation(eigenMat, fr, flr):
     
    sigma = np.zeros((np.shape(fr)[1],1))
    for i in range(np.shape(sigma)[0]):
        sigma[i] = (fr[:,i]/flr[:,i])**2
    bf = np.sqrt(sigma/np.shape(fr)[0])
    w = np.zeros((np.shape(fr)[0],))
    for i in range(np.shape(w)[0]):
        w[i] = 0.5**((np.shape(w)[0]-i-1)/np.shape(w)[0])
    w = w / np.sum(w)
    i = 0
    lamda = 0
    for bft in bf:
        lamda = lamda + bft**2*w[i]
        i += 1
    newMatrix = lamda * eigenMat

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
def structure(neweyWestMat, r, exposure, omiga, h = 252, E0 = 1.05):

    n = len(r)

    Q3 = 3/4 * r
    Q1 = 1/4 * r
    sigmaReference = 1/1.35 * (Q3 -Q1)
    zu = np.abs(np.diag(omiga).reshape(n,1)/sigmaReference - 1)
    gama = np.zeros((n,1)) 
    for i in range(len(zu)):
        gama[i,0] = (min(1, max(0, h/120-0.5))) * (min(1, max(0, np.exp(1-zu[i,0])))) 
    
    locations = np.where(gama == 1)[0]
    locations2 = np.where(gama != 0)[0]
    sigmaStruct = np.zeros((n,n))
    lnSigma = list()
    for location in locations:
        sigmaStruct[location,location] = neweyWestMat[location,location]
        lnSigma.append(neweyWestMat[location,location])
    lnSigma = np.log(np.mat(lnSigma))
    # use generalized linear model
    model = sm.GLM(lnSigma, exposure)
    b = model.fit().params
    SigmaXb = np.dot(exposure, b.T)
    for location in locations2:
        sigmaStruct[location,location] = E0 * np.exp(SigmaXb[location,0])

    newMatrix = gama * neweyWestMat + (1 - gama) * sigmaStruct

    return newMatrix

# bayes adjustment
# sigmaSHn = vn * sigmaSn + (1-vn)*sigman  n = 1,2,3,....,N
# sigmaSn = Sigma(wn*sigman)  n = 1,2,3,......, N
# vn = 1 /(deltaSn/(q*abs(sigman - sigmaSn)) + 1)
# deltaSn = sqrt(Sigma((sigman - sigmaSn)^2)/NSn)
# weight is read from barra_factor_exposure also weight should be normalized beforehand
def bayes(structureMat, weight, groups = 10, q = 1):
    
    sigma = np.mat(np.diag(structureMat)).T
    
    dfw = pd.DataFrame({'sigma':np.diag(structureMat),'weight':weight})
    dfw['location'] = dfw.index
    dfw.sort_values(by = 'weight', ascending = True, inplace = True)
    dfw.reset_index(drop = True, inplace = True)
    v = np.zeros((len(dfw)//groups,1))
    sigmaS = np.zeros((len(dfw)//groups,1))
    for i in range(groups-1):
        df = dfw.iloc[i*(len(dfw)//groups),(i+1)*(len(dfw)//groups)]
        locations = list(df['location'])
        sigmaTmp = sigma[locations]
        sigmaSTmp = np.ones((len(locations),1))*((df['weight']*df['sigma']).mean())
        deltaS = df['sigma'].std()
        sigmaS[locations] = sigmaSTmp
        v[locations] = 1 /(deltaS/(q*abs(sigmaTmp - sigmaSTmp)) + 1)
    
    df = dfw.iloc[(groups-1)*(len(dfw)//groups),len(dfw)]
    locations = list(df['location'])
    sigmaTmp = sigma[locations]
    sigmaSTmp = np.ones((len(locations),1))*((df['weight']*df['sigma']).mean())
    deltaS = df['sigma'].std()
    sigmaS[locations] = sigmaSTmp
    v[locations] = 1 /(deltaS/(q*abs(sigmaTmp - sigmaSTmp)) + 1)

    newArray = v*sigmaS + (1-v)*sigma
    newMatrix = np.eye(len(sigma))
    for i in range(len(sigma)):
        newMatrix[i,i] = newArray[i,0]

    return newMatrix

