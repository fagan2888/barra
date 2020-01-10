
import numpy as np
import pandas as pd
import random
import math
from scipy.stats import pearsonr

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
def neweyWestRaw(fr, q):
    
    cov0 = np.zeros((np.shape(fr)[0],np.shape(fr)[0]))
    t = np.shape(fr)[1]
    for i in range (t):
        cov0 = cov0 + np.dot(fr[:,i],fr[:,i].T)
    cov0 = cov0/t

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

    newMatrix = exponentMat + sigmaCov

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


#####################################
##############reference##############

H_L_vol = 84  # factor volatility half life
H_L_corr = 252  # factor volatility half life
H_L_VRA = 42  # factor
H_L_special = 84
H_L_special_NW = 252
H_L_special_VRA = 42
Lags_vol = 5  # Newey-West volatility lags
Lags_corr = 2  # Newey-West volatility correlation
Lags_special = 5
H_window = 252  # 过去252个交易日的数据，最好使用过去两年的交易数据，因为H_L_corr=252
F_vol_raw = pd.DataFrame()
F_corr_raw = pd.DataFrame()
Predict_period = 21  # 对未来21天的风险进行预测，若低于21可能会出现问题，Newey_West处理所导致
adj_coef = 1.2  # yn_eig_risk_adj函数 方正证券取值1.2 Barra取值1.4
M = 1000  # 模拟次数
E_0 = 1.05  # E_0是一个略大于1的常数，用于去除残差项的指数次幂带来的偏误


def yn_f_raw(data, half_life = H_L_corr):  # type of data is np.array
    fun_data = data.copy()
    lambda_t = 0.5 ** (1 / half_life)
    weights_t = lambda_t ** (np.arange(fun_data.shape[0] - 1, -1, -1))
    weights_t = weights_t / weights_t.sum()
    f_raw = np.cov(fun_data, rowvar=False, aweights=weights_t, ddof=0)  # ddof=0: 样本方差 /n 默认为: ddof=1
    return f_raw

def yn_newey_west(data, half_life = H_L_corr, lags = Lags_corr):
    fun_data = data.copy()
    lambda_t = 0.5 ** (1 / half_life)
    c_newey_west = np.zeros((fun_data.shape[1], fun_data.shape[1]))
    for i_lag in range(1, lags+1):
        c_newey_west_t = np.zeros((fun_data.shape[1], fun_data.shape[1]))
        for j_factor in range(fun_data.shape[1]):
            fun_data_t = fun_data.copy()  # Notice: use copy
            fun_data_j = fun_data_t[:-i_lag, j_factor]
            fun_data_t = fun_data_t[i_lag:, :]
            weights_t = lambda_t ** (np.arange(fun_data_t.shape[0] - 1, -1, -1))
            weights_t = weights_t / weights_t.sum()
            volatility_t = np.cov(fun_data_t, fun_data_j, rowvar=False, aweights=weights_t, ddof=0)
            c_newey_west_t[:, j_factor] = volatility_t[:-1, -1]
        coef_t = 1 - (i_lag / (lags + 1))
        c_newey_west = c_newey_west + coef_t * (c_newey_west_t + c_newey_west_t.T)
    return c_newey_west

def yn_eig_risk_adj(data):  # 会使用全局变量: M H_window adj_coef
    f_nw = data.copy()
    w_0, u_0 = np.linalg.eig(f_nw)
    d_0 = np.mat(u_0.T) * np.mat(f_nw) * np.mat(u_0)
    m_volatility = np.zeros((M, f_nw.shape[0]))
    # 模拟M次
    for m in range(M):
        b_m = np.zeros((f_nw.shape[0], H_window))  # N*T
        for i_row in range(b_m.shape[0]):
            b_m[i_row, :] = np.random.normal(loc=0, scale=np.sqrt(d_0[i_row, i_row]), size=H_window)
        r_m = np.mat(u_0) * np.mat(b_m)
        r_m = np.array(r_m.T)  # notice: 转置处理
        f_nw_m = np.cov(r_m, rowvar=False, ddof=0)  # 不需要Weights
        w_m, u_m = np.linalg.eig(f_nw_m)
        d_m = np.mat(u_m.T) * np.mat(f_nw_m) * np.mat(u_m)
        d_m_real = np.mat(u_m.T) * np.mat(f_nw) * np.mat(u_m)
        m_volatility[m, :] = np.diag(d_m_real) / np.diag(d_m)
    gamma_t = np.sqrt(m_volatility.mean(axis=0))
    gamma_t = adj_coef * (gamma_t - 1) + 1
    return gamma_t

def yn_vol_regime_adj(data):  # 会使用全局变量: H_L_VRA
    fun_data = data[:-1]
    fun_data = fun_data[~np.isnan(fun_data)]
    if len(fun_data) < H_L_VRA:
        return np.nan
    else:
        lambda_t = 0.5 ** (1 / H_L_VRA)
        weights_t = lambda_t ** (np.arange(fun_data.shape[0] - 1, -1, -1))
        weights_t = weights_t / weights_t.sum()
        lambda_f_var = np.dot(fun_data ** 2, weights_t)
        lambda_f = np.sqrt(lambda_f_var)
        return lambda_f

