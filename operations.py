
import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
import numpy as np
import sys
sys.path.extend('../')
from db import database
from db.config import *

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
def factorExposure(date, industryFactors, stocks = None):

    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('factor_exposure_barra_20200107', meta, autoload = True)
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
        #t.c.weight, # need further treatment
    ]
    sql = select(columns)
    sql = sql.where(t.c.trade_date == date)
    if stocks != None:
        sql = sql.where(t.c.stock_id.in_(stocks))
    exposure = pd.read_sql(sql, db)

    w = exposure
    w['country'] = 1
    
    for industry in industryFactors:
        w[industry] = 0
    for i in range(len(w)):
        w['industry_'+str(w['industry'][i])][i] = 1
    
    w = w.drop('industry', axis = 1)

    return w

# calculate factor fluctuation ratio
def FactorFluctuation(factorReturn):
    
    flr = factorReturn.std()

    return flr

# calculate covariance matirx
def FactorCovariance(w, sigma, omiga):

    covarianceMatrix = np.dot(np.dot(w,sigma),w.T) + omiga

    return covarianceMatrix

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
def saveFactorCovariance(mat, names, date, sdate = None, edate = None):

    df = pd.DataFrame(columns = ['bf_date','bf_factor1','bf_factor2','bf_cov'])
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
    df['bf_factor1'] = dfFactor1
    df['bf_factor2'] = dfFactor2
    df['bf_cov'] = covij
    df['bf_date'] = date
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
    if sdate != None:
        sql = sql.where(t.c.bf_date >= sdate)
    if edate != None:
        sql = sql.where(t.c.bf_date <= edate)
    dfBase = pd.read_sql(sql, db)
    dfBase['bf_date'] = dfBase['bf_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['bf_date','bf_factor1','bf_factor2'], inplace = True)

    database.batch(db, t, df, dfBase, timestamp = False)

# save stock return covariance into barra_covariance
def saveStockCovariance(mat, names, date, sdate = None, edate = None):

    df = pd.DataFrame(columns = ['bc_date','bc_stock1','bc_stock2','bc_cov'])
    dfStock1 = list()
    dfStock2 = list()
    covij = list()
    i = 0
    for name1 in names:
        j = i
        for name2 in names[i:]:
            dfStock1.append(name1)
            dfStock2.append(name2)
            covij.append(mat[i,j])
            j += 1
        i += 1
    df['bc_stock1'] = dfStock1
    df['bc_stock2'] = dfStock2
    df['bc_cov'] = covij
    df['bc_date'] = date
    df['bc_date'] = df['bc_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    df.set_index(['bc_date','bc_stock1','bc_stock2'], inplace = True)

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
    if sdate != None:
        sql = sql.where(t.c.bc_date >= sdate)
    if edate != None:
        sql = sql.where(t.c.bc_date <= edate)
    dfBase = pd.read_sql(sql, db)
    dfBase['bc_date'] = dfBase['bc_date'].map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    dfBase.set_index(['bc_date','bc_stock1','bc_stock2'], inplace = True)

    database.batch(db, t, df, dfBase, timestamp = False)

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


