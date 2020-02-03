
import numpy as np
import pandas as pd
from sqlalchemy import *
from db.config import *
from db import database
from db.factors import styleFactors, industryFactors
from datetime import datetime, timedelta, date
import statsmodels.api as sm
from ipdb import set_trace
import click


@click.group(invoke_without_command = True)
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime(2019,12,31)-timedelta(days = 5)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime(2019,12,31)).strftime('%Y-%m-%d'),help = 'end date')
@click.pass_context
def cal(ctx, sdate, edate):
    ctx.invoke(handle, sdate = sdate, edate = edate)


@cal.command()
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime(2019,12,31)-timedelta(days = 5)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime(2019,12,31)).strftime('%Y-%m-%d'),help = 'end date')
@click.pass_context
# work out some basic data for barra
def handle(ctx, sdate, edate):
    
    #
    styleFactors.sort()
    industryFactors.sort()
    
    # load factor exposures of every stocks
    db = create_engine(uris['factor'])
    sql = "select * from `factor_exposure_barra_20200107` where trade_date >= '" + sdate + "' and trade_date <='" + edate + "'"
    dfExposure = pd.read_sql(sql, db)
    if len(dfExposure) == 0:
        print('no exposure data! please change sdate and edate!')
        exit()

    # load daily returns of every stocks
    db = create_engine(uris['wind'])
    meta = MetaData(bind = db)
    t = Table('ashareeodprices', meta, autoload = True)
    columns = [
        t.c.S_INFO_WINDCODE,
        t.c.TRADE_DT,
        t.c.S_DQ_ADJCLOSE
    ]
    sql = select(columns)
    #sql = sql.where(t.c.S_DQ_TRADESTATUS != '停牌').where(t.c.S_DQ_TRADESTATUS != '待核查')
    sql = sql.where(t.c.TRADE_DT <= pd.Timestamp(edate).strftime('%Y%m%d'))
    sql = sql.where(t.c.TRADE_DT >= pd.Timestamp(datetime.strptime(sdate,'%Y-%m-%d') - timedelta(days = 100)).strftime('%Y%m%d'))
    dfAdjClose = pd.read_sql(sql, db)
    
    # it is necessary to make sure that stocks are both included in exposure table and wind table
    stocks = set(dfExposure['stock_id']).intersection(set(dfAdjClose['S_INFO_WINDCODE']))
    dfExposure = dfExposure[dfExposure['stock_id'].isin(stocks)]
    dfExposureG = dfExposure.groupby('trade_date')
    
    dfAdjClose = dfAdjClose[dfAdjClose['S_INFO_WINDCODE'].isin(stocks)]
    dfAdjCloseG = dfAdjClose.groupby('S_INFO_WINDCODE')
    dfAdjClose = pd.DataFrame(columns = ['pct_change', 'S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE'])
    dfList = []
    for stock in stocks:
        dfTmp = dfAdjCloseG.get_group(stock).copy()
        dfTmp.sort_values(by = 'TRADE_DT', ascending = True, inplace = True)
        dfTmp.reset_index(inplace = True, drop = True)
        pct_change = dfTmp['S_DQ_ADJCLOSE'].pct_change()
        dfTmp.insert(0, 'pct_change', pct_change)
        dfTmp = dfTmp.fillna(0)
        dfList.append(dfTmp)
    dfAdjClose = pd.concat(dfList , axis = 0, sort = True)
    dfAdjClose.drop_duplicates(['TRADE_DT','S_INFO_WINDCODE'], inplace = True)
    dfAdjCloseG = dfAdjClose.groupby('TRADE_DT')        

    # main part
    dfResid = pd.DataFrame(columns = ['trade_date','stock_id','resid'])
    dfParams = pd.DataFrame(columns = ['trade_date'] + ['country'] + styleFactors + industryFactors)
    dfParams.set_index('trade_date', inplace = True)
    dfPList = []
    dfRList = []
    # rn = fc + Sigma(Xi*fi) + Sigma(Xs*fs) + un  Sigma(w*fi) = 0  un is resid
    for date, exposure in dfExposureG:
        windCodes = list(exposure['stock_id'])
        dateWind = pd.Timestamp(date).strftime('%Y%m%d')
        dfAdjClose = dfAdjCloseG.get_group(dateWind).copy()
        dfAdjClose = dfAdjClose[dfAdjClose['S_INFO_WINDCODE'].isin(windCodes)]
        dfAdjClose = dfAdjClose.fillna(0)
        exposure.sort_values(by = 'stock_id', inplace = True)
        exposure = exposure.fillna(0)

        r = np.matrix(dfAdjClose.sort_values('S_INFO_WINDCODE')['pct_change']).T
        # exposures of country factor
        Xc = np.ones((len(exposure),1))
        # exposures of style factor
        Xs = np.matrix(exposure[styleFactors])
        # exposures of industry factor
        Xi = np.matrix(pd.get_dummies(exposure['industry']).sort_index(axis = 1))
        X = np.hstack((Xc,Xs,Xi))
        w = ((Xi.T) * (np.matrix(exposure['weight']).T)) / (exposure['weight'].sum())
        w = np.array(w).reshape(len(w),)
        # use generalized linear model
        model = sm.GLM(r,X, var_weights = np.sqrt(exposure['weight'].values))
        Q = np.hstack([[0],np.zeros(len(styleFactors)),w])
        result = model.fit_constrained((Q, 0.0))
        params = result.params
        resid = result.resid_response

        # industry changes.
        # sometimes new industries are added sometimes old industires are deleted
        # we only care about industries in industryList
        industryList = list(set(exposure['industry']))
        industryList.sort()
        factors = ['country'] + styleFactors + industryFactors
        dfP = pd.DataFrame(columns = ['trade_date'] + factors)
        dfP.set_index('trade_date', inplace =True)
        for i in range (1 + len(styleFactors)):
            dfP.loc[date,factors[i]] = params[i]
        k = 1+len(styleFactors)
        for ind in industryList:
            dfP.loc[date,'industry_'+ind] = params[k]
            k += 1
        dfP = dfP.fillna(0)
        dfPList.append(dfP)
        
        dfR = pd.DataFrame(columns = ['trade_date','stock_id','resid'])
        dfR['stock_id'] = exposure['stock_id'] 
        dfR['resid'] = resid
        dfR['trade_date'] = date
        dfRList.append(dfR)
        
    dfParams = pd.concat(dfPList, axis = 0)
    dfResid = pd.concat(dfRList, axis = 0)
    dfParams.sort_index(axis = 1, inplace = True)
    # connect to database and update factor returns
    db = create_engine(uris['factor'])
    meta = MetaData(bind = db)
    t = Table('barra_factor_return', meta, autoload = True)
    sql = "select trade_date, " + ','.join(dfParams.columns.values) + " from `barra_factor_return` where trade_date >= '" + sdate + "' and trade_date <='" + edate +"'"
    dfBase = pd.read_sql(sql, db)
    dfBase.sort_index(axis = 1, inplace = True)
    dfBase.set_index('trade_date', inplace = True)

    database.batch(db,t,dfParams,dfBase,timestamp = False)
    print('factor return updated!')
    
    dfResid.set_index(['trade_date','stock_id'], inplace = True)
    # connect to database and update regression resids
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
    dfBase = pd.read_sql(sql, db)
    dfBase.set_index(['trade_date','stock_id'], inplace = True)
    database.batch(db,t,dfResid,dfBase,timestamp = False)
    print('regression reside updated!')


if __name__ == '__main__':
    edate = datetime.today()
    edate = date(2019,12,31)
    sdate = pd.Timestamp(edate-timedelta(days = 5)).strftime('%Y-%m-%d')
    edate = pd.Timestamp(edate).strftime('%Y-%m-%d')
    handle(sdate, edate)
