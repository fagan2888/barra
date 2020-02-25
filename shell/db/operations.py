
import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
import numpy as np
import sys
sys.path.extend('../')
from db import database
from db.config import *
from ipdb import set_trace

# load factor return 
def factorReturn(sdate, edate):

    db = create_engine(uris['factor'])
    sql = "select * from `barra_factor_return` where trade_date >= '" + sdate + "' and trade_date <='" + edate +"'"
    payback = pd.read_sql(sql, db)

    return payback

# load regression resid for every stock or some stocks
def regressionResid(sdate, edate, stocks = None):

    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_regression_resid', meta, autoload = True)
    columns = [
        t.c.trade_date,
        t.c.stock_id,
        t.c.resid,
    ]
    sql = select(columns)
    sql = sql.where(t.c.trade_date >= sdate)
    sql = sql.where(t.c.trade_date <= edate)
    if stocks != None:
        sql = sql.where(t.c.stock_id.in_(stocks))
    resid = pd.read_sql(sql, db)

    return resid

# load factor exposure of every stock or some stocks
def factorExposure(dates, industryFactors, stocks = None):

    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('factor_exposure_barra_20200220', meta, autoload = True)
    columns = [
        t.c.trade_date,
        t.c.stock_id,
        #t.c.country,
        t.c.volatility,
        t.c.dividend_yield,
        t.c.quality,
        t.c.momentum,
        t.c.short_term_reverse,
        t.c.value,
        t.c.linear_size,
        t.c.nonlinear_size,
        t.c.growth,
        t.c.liquidity,
        t.c.sentiment,
        t.c.industry, # need further treatment
        t.c.weight, # need further treatment
    ]
    sql = select(columns)
    sql = sql.where(t.c.trade_date.in_(dates))
    if stocks != None:
        sql = sql.where(t.c.stock_id.in_(stocks))
    exposure = pd.read_sql(sql, db)

    w = exposure
    w['country'] = 1
    
    industries = pd.get_dummies(w.industry)
    industries.columns = ['industry_'+name for name in list(industries.columns)]
    w = w.drop('industry', axis = 1)
    w = pd.merge(w,industries, left_index = True, right_index = True)

    return w

# calculate factor fluctuation ratio
def FactorFluctuation(factorReturn):
    
    flr = factorReturn.std()

    return flr

# calculate covariance matirx
def FactorCovariance(Exposure, Sigma, Omiga, dates):

    dates.sort()
    i = 0
    dfList = []
    for date in dates:
        stocks = list(Omiga[i].columns)
        stocks.sort()
        exposure = Exposure[Exposure.index == date]
        exposure = exposure[exposure['stock_id'].isin(stocks)]
        w = np.mat(exposure.set_index('stock_id').sort_index(axis = 0, ascending = True))
        sigma = np.mat(Sigma[i,:,:].reshape(np.shape(Sigma)[1:3]))
        omiga = np.mat(Omiga[i].sort_index(axis = 1))
        covarianceMatrix = np.dot(np.dot(w,sigma),w.T) + omiga
        dfTmp = pd.DataFrame(covarianceMatrix)
        dfTmp.columns = stocks
        dfList.append(dfTmp)
        i += 1

    return dfList

# calculate stock fluctuation ratio
def stockFluctuation(stockResid):

    stockFlr = stockResid.std(axis = 0)

    return stockFlr

# save factor return fluctuation ratio into barra_fluctuation_ratio
def saveFlr(df, dates):

    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_factor_fluctuation_ratio', meta, autoload = True)
    columns = [
        t.c.bf_date,
        t.c.bf_factor,
        t.c.bf_flr,
    ]
    sql = select(columns)
    sql = sql.where(t.c.bf_date.in_(dates))
    dfBase = pd.read_sql(sql, db)
    dfBase['bf_date'] = dfBase['bf_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['bf_date','bf_factor'], inplace = True)

    database.batch(db, t, df, dfBase, timestamp = False)

# save factor return fluctuation ratio into barra_fluctuation_ratio
def saveStockFlr(df, dates):

    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_resid_fluctuation_ratio', meta, autoload = True)
    columns = [
        t.c.br_date,
        t.c.br_stock_id,
        t.c.br_flr,
    ]
    sql = select(columns)
    sql = sql.where(t.c.br_date.in_(dates))
    dfBase = pd.read_sql(sql, db)
    dfBase['br_date'] = dfBase['br_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['br_date','br_stock_id'], inplace = True)

    database.batch(db, t, df, dfBase, timestamp = False)


# save factor return covariance into barra_factor_covariance
def saveFactorCovariance(matAll, names, dates):

    dates.sort()
    factorNum = np.shape(matAll)[1]
    dfList = []
    k = 0
    for date in dates:
        mat = matAll[k,:,:].reshape(factorNum, factorNum)
        dfTmp = pd.DataFrame(columns = ['bf_date','bf_factor1','bf_factor2','bf_cov'])
        dfFactor1 = list()
        dfFactor2 = list()
        covij = list()
        i = 0
        for name1 in names:
            j = i
            for name2 in names[i:]:
                dfFactor1.append(name1)
                dfFactor2.append(name2)
                covij.append(mat[i,j])
                j += 1
            i += 1
        dfTmp['bf_factor1'] = dfFactor1
        dfTmp['bf_factor2'] = dfFactor2
        dfTmp['bf_cov'] = covij
        dfTmp['bf_date'] = date
        dfList.append(dfTmp)
        k += 1
    df = pd.concat(dfList, axis = 0)
    df['bf_date'] = df['bf_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    df.set_index(['bf_date','bf_factor1','bf_factor2'], inplace = True)
    
    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_factor_covariance', meta, autoload = True)
    columns = [
        t.c.bf_date,
        t.c.bf_factor1,
        t.c.bf_factor2,
        t.c.bf_cov,
    ]
    sql = select(columns)
    sql = sql.where(t.c.bf_date.in_(dates))
    dfBase = pd.read_sql(sql, db)
    dfBase['bf_date'] = dfBase['bf_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['bf_date','bf_factor1','bf_factor2'], inplace = True)

    database.batch(db, t, df, dfBase, timestamp = False)

# save stock return variance into barra_stock_variance
def saveStockVariance(dfAll, dates):

    dates.sort()
    n = len(dfAll)
    dfList = []
    for i in range(n):
        dfTmp = pd.DataFrame(columns = ['bs_date','bs_stock_id','bs_var']) 
        stocks = list()
        stockVars = list()
        k = 0
        for column in dfAll[i].columns:
            stockVars.append(dfAll[i].iloc[k,k])
            stocks.append(column)
            k += 1
        dfTmp['bs_stock_id'] = stocks
        dfTmp['bs_var'] = stockVars
        dfTmp['bs_date'] = dates[i]
        dfList.append(dfTmp)
    df = pd.concat(dfList, axis = 0)
    df['bs_date'] = df['bs_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    df.set_index(['bs_date','bs_stock_id'], inplace = True)
    
    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_stock_variance', meta, autoload = True)
    columns = [
        t.c.bs_date,
        t.c.bs_stock_id,
        t.c.bs_var,
    ]
    sql = select(columns)
    sql = sql.where(t.c.bs_date.in_(dates))
    dfBase = pd.read_sql(sql, db)
    dfBase['bs_date'] = dfBase['bs_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['bs_date','bs_stock_id'], inplace = True)

    database.batch(db, t, df, dfBase, timestamp = False)

# save stock return covariance into barra_covariance
def saveStockCovariance(dfAll, dates):

    dates.sort()
    dfList = []
    k = 0
    for date in dates:
        df = pd.DataFrame(columns = ['bc_date','bc_stock1','bc_stock2','bc_cov'])
        names = list(dfAll[k].columns)
        dfStock1 = list()
        dfStock2 = list()
        covij = list()
        i = 0
        for name1 in names:
            j = i
            for name2 in names[i:]:
                dfStock1.append(name1)
                dfStock2.append(name2)
                covij.append(dfAll[k].iloc[i,j])
                j += 1
            i += 1
        df['bc_stock1'] = dfStock1
        df['bc_stock2'] = dfStock2
        df['bc_cov'] = covij
        df['bc_date'] = date
        df['bc_date'] = df['bc_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
        df.set_index(['bc_date','bc_stock1','bc_stock2'], inplace = True)
        dfList.append(df)
        k += 1
    dfNew = pd.concat(dfList, axis = 0)

    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_covariance', meta, autoload = True)
    columns = [
        t.c.bc_date,
        t.c.bc_stock1,
        t.c.bc_stock2,
        t.c.bc_cov,
    ]
    sql = select(columns)
    sql = sql.where(t.c.bc_date.in_(dates))
    dfBase = pd.read_sql(sql, db)
    dfBase['bc_date'] = dfBase['bc_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['bc_date','bc_stock1','bc_stock2'], inplace = True)

    database.batch(db, t, dfNew, dfBase, timestamp = False)

# load factor return 
def loadFlr(sdate, edate):

    db = create_engine(uris['factor'])
    sql = "select * from `barra_factor_fluctuation_ratio` where bf_date >= '" + sdate + "' and bf_date <='" + edate +"'"
    flr = pd.read_sql(sql, db)

    return flr

# load factor return 
def loadResidFlr(sdate, edate):

    db = create_engine(uris['factor'])
    sql = "select * from `barra_resid_fluctuation_ratio` where br_date >= '" + sdate + "' and br_date <='" + edate +"'"
    residFlr = pd.read_sql(sql, db)

    return residFlr


