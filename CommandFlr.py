

import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
import numpy as np
from db.config import *
from db import database
from db.operations import *
from ipdb import set_trace
import click


@click.group(invoke_without_command = True)
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime(2018,1,2)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime(2020,1,21)).strftime('%Y-%m-%d'), help = 'end date')
@click.option('--delta', 'delta', default = 21, help = 'use data of delta dates to predict fluctuation ratio')
@click.pass_context
def cov(ctx, sdate, edate, delta):
    ctx.invoke(handle, sdate = sdate, edate = edate, delta = delta)


@cov.command()
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime(2018,1,2)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime(2020,1,21)).strftime('%Y-%m-%d'), help = 'end date')
@click.option('--delta', 'delta', default = 21, help = 'use data of delta dates to predict fluctuation ratio')
@click.pass_context
# main function
def handle(ctx, sdate, edate, delta):
    
    fr = factorReturn(sdate, edate)
    fr['trade_date'] = fr['trade_date'].map(lambda x: x.strftime('%Y-%m-%d'))
    dates = list(fr['trade_date'])
    dates.sort()
    fr.set_index('trade_date', inplace = True)
    fr.sort_index(ascending = True, inplace = True)
    fr.sort_index(axis = 1, inplace = True)
    fr = fr.fillna(0)
    factorNames = list(fr.columns)
  
    dfList = []
    for i in range(delta+1,len(fr)):
        frTmp = fr.iloc[i-delta:i-1]
        flr = FactorFluctuation(frTmp)
        dfTmp = pd.DataFrame(columns = ['bf_date','bf_factor','bf_flr'])
        dfTmp['bf_factor'] = flr.index
        dfTmp['bf_flr'] = flr.values
        dfTmp['bf_date'] = dates[i]
        dfList.append(dfTmp)
    
    df = pd.concat(dfList,axis = 0)
    df.set_index(['bf_date','bf_factor'], inplace = True)
    saveFlr(df = df, dates = dates[delta+1:])
    print('Factor return fluctucation saved! Check barra_factor_fluctucation_ratio to see more details.')
    resid = regressionResid(sdate, edate)
    stocks = set(resid['stock_id'])
    resid.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    resid.set_index(['trade_date','stock_id'], inplace = True)
    resid = resid.unstack()['resid'].fillna(0)
    stockIds = list(resid.columns)
  
    dfList = []
    for i in range(delta+1,len(resid)):
        residTmp = resid.iloc[i-delta:i-1]
        stockFlr = stockFluctuation(residTmp)
        dfTmp = pd.DataFrame(columns = ['br_date','br_stock_id','br_flr'])
        dfTmp['br_stock_id'] = stockFlr.index
        dfTmp['br_flr'] = stockFlr.values
        dfTmp['br_date'] = dates[i]
        dfList.append(dfTmp)
   
    df = pd.concat(dfList,axis = 0)
    df.set_index(['br_date','br_stock_id'], inplace = True)
    saveStockFlr(df, dates[delta+1:])
    print('Stock return resid fluctucation saved! Check barra_resid_fluctucation_ratio to see more details.')
 

if __name__ == '__main__':
    sdate = pd.Timestamp(datetime(2018,1,2)).strftime('%Y-%m-%d')
    edate = pd.Timestamp(datetime(2020,1,21)).strftime('%Y-%m-%d')
    delta = 21
    handle(sdate, edate, delta)

