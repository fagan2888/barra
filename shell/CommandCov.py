

import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
import numpy as np
from config import *
from db.factors import styleFactors, industryFactors
from CommandMatrixAdjust import *
from optparse import OptionParser
from ipdb import set_trace

# payback and resid is calculated in CommandCal.py
industryFactors.sort()
 
# load factor return 
def factorReturn(sdate, edate):

    db = create_engine(uris['multi_factor'])
    sql = "select * from `barra_factor_return` where trade_date >= '" + sdate + "' and trade_date <='" + edate +"'"
    payback = pd.read_sql(sql, db)

    return payback

# load regression resid for every stock or some stocks
def regressionResid(sdate, edate, stocks = None):

    db = create_engine(uris['multi_factor'])
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

    db = create_engine(uris['multi_factor'])
    meta = MetaData(bind = db)
    t = Table('factor_exposure_barra', meta, autoload = True)
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

# calculate factor fluctuation rate
def FactorFluctuation(factorReturn):
    
    flr = factorReturn.std()

    return flr

# calculate covariance matirx
def FactorCovariance(w, sigma, omiga):

    covarianceMatrix = np.dot(np.dot(w,sigma),w.T) + omiga

    return covarianceMatrix

# main function
def handle(sdate, edate, date):
    
    fr = factorReturn(sdate, edate)
    fr.set_index('trade_date', inplace = True)
    fr.sort_index(ascending = True, inplace = True)
    fr.sort_index(axis = 1, inplace = True)
    fr = fr.fillna(0)
   
    flr = FactorFluctuation(fr)
    print('fluctuation rate of every factors are as folllows:')
    print(flr)
    
    resid = regressionResid(sdate, edate)
    stocks = set(resid['stock_id'])
    resid.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    resid.set_index(['trade_date','stock_id'], inplace = True)
    resid = resid.unstack()['resid'].fillna(0)
   
    weight = factorExposure(date, industryFactors, stocks)
    weight.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    weight.set_index(['trade_date','stock_id'], inplace = True)
    weight.sort_index(axis = 1 , inplace = True)
    w = np.matrix(weight)
    w = nothing(w)
    
    sigma = np.cov(np.matrix(fr).T)
    omiga = np.diag(resid.apply(lambda x: x**2).mean())
    covarianceMatrix = FactorCovariance(w, sigma, omiga)
    
    print('covarianceMatrix of is')
    print(covarianceMatrix)
    print('task finished!')


if __name__ == '__main__':
    opt = OptionParser()
    endDate = pd.Timestamp(datetime.today()).strftime('%Y-%m-%d')
    startDate = str(int(endDate[0:4])-1)+'-01-01'
    defaultDate = pd.Timestamp(datetime.today() - timedelta(days = 1)).strftime('%Y-%m-%d')
    defaultDate = '2019-12-31'
    opt.add_option('-s','--sdate',help = 'start date', default = startDate)
    opt.add_option('-e','--edate',help = 'end date', default = endDate)
    opt.add_option('-d','--date', help = 'date', default = defaultDate)
    opt, arg = opt.parse_args()
    handle(opt.sdate, opt.edate, opt.date)

