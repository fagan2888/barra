

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
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime.today()- timedelta(days = 5)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime.today()).strftime('%Y-%m-%d'), help = 'end date')
@click.option('--delta', 'delta', default = 21, help = 'use data of delta dates to predict fluctuation ratio')
@click.option('--ratio', 'ratio', default = 0.10, help = 'when calculating resid fluctuation, stocks for which "len(data)/timeWindow" is less than "ratio" will not be considered')
@click.pass_context
def flr(ctx, sdate, edate, delta, ratio):
    ctx.invoke(handle, sdate = sdate, edate = edate, delta = delta, ratio = ratio)


@flr.command()
# main function
def handle(sdate, edate, delta, ratio):
    
    sdate = pd.Timestamp(sdate).strftime('%Y-%m-%d')
    edate = pd.Timestamp(edate).strftime('%Y-%m-%d')
    #
    startDate = pd.Timestamp(datetime.strptime(sdate,'%Y-%m-%d') - timedelta(days = delta+100)).strftime('%Y-%m-%d')
    fr = factorReturn(startDate, edate)
    fr['trade_date'] = fr['trade_date'].map(lambda x: x.strftime('%Y-%m-%d'))

    if len(fr[fr.trade_date >= sdate]) == 0:
        print('no factor return data between sdate and edate! please check factor return table and change sdate/edate')
        exit()

    dates = list(set(fr['trade_date']))
    dates.sort()
    datesHelper = [t for t in dates if t >= sdate]
    if len(dates) < delta:
        print('data is not enough for fitting time window. please check factor return table and change sdate/edate. or you can change time window')
        exit()
    if len(dates)-len(datesHelper) < (delta-1):
        datesHelper = dates[delta-1:len(dates)]
    dates = dates[len(dates)- len(datesHelper)-(delta-1):len(dates)]

    fr = fr[fr.trade_date.isin(dates)]
    fr.set_index('trade_date', inplace = True)
    fr.sort_index(ascending = True, inplace = True)
    fr.sort_index(axis = 1, inplace = True)
  
    dfList = []
    for i in range(delta-1,len(fr)):
        frTmp = fr.iloc[i-delta+1:i+1]
        flr = FactorFluctuation(frTmp)
        dfTmp = pd.DataFrame(columns = ['bf_date','bf_factor','bf_flr'])
        dfTmp['bf_factor'] = flr.index
        dfTmp['bf_flr'] = flr.values
        dfTmp['bf_date'] = dates[i]
        dfList.append(dfTmp)
    
    df = pd.concat(dfList,axis = 0)
    df.set_index(['bf_date','bf_factor'], inplace = True)
    saveFlr(df = df, dates = dates[delta-1:])
    print('Factor return fluctucation saved! Check barra_factor_fluctucation_ratio to see more details.')
    
    resid = regressionResid(startDate, edate)
    resid['trade_date'] = resid['trade_date'].map(lambda x: x.strftime('%Y-%m-%d'))
    resid = resid[resid.trade_date.isin(dates)]
    resid.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    resid.set_index(['trade_date','stock_id'], inplace = True)
    resid = resid.unstack()['resid']
    stockIds = list(resid.columns)
  
    dfList = []
    for i in range(delta-1,len(resid)):
        residTmp = resid.iloc[i-delta+1:i+1]
        mask = residTmp.isna().sum()/delta
        for col in stockIds:
            if mask[col] > ratio:
                residTmp = residTmp.drop(col, axis = 1)
        stockFlr = stockFluctuation(residTmp)
        dfTmp = pd.DataFrame(columns = ['br_date','br_stock_id','br_flr'])
        dfTmp['br_stock_id'] = stockFlr.index
        dfTmp['br_flr'] = stockFlr.values
        dfTmp['br_date'] = dates[i]
        dfList.append(dfTmp)
   
    df = pd.concat(dfList,axis = 0)
    df.set_index(['br_date','br_stock_id'], inplace = True)
    saveStockFlr(df, dates[delta-1:])
    print('Stock return resid fluctucation saved! Check barra_resid_fluctucation_ratio to see more details.')
