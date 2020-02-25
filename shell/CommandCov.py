
import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
from time import strptime
import numpy as np
from db.config import *
from db.factors import styleFactors, industryFactors
from db import database
from db.operations import *
from db.matrixAdjust import *
from ipdb import set_trace
import click

# payback and resid is calculated in CommandCal.py
styleFactors.sort()
industryFactors.sort()

@click.group(invoke_without_command = True)
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime.today()-timedelta(days = 5)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime.today()).strftime('%Y-%m-%d'), help = 'end date')
@click.option('--delta', 'delta', default = 250, help = 'suppose that the covariance of one certain day can be affected by the data of past delta days')
@click.option('--deltaEigen', 'deltaEigen', default = 250, help = 'time windows for eigen adjustment')
@click.option('--cov','cov', default = False, help = 'update stock return covariance or not')
@click.pass_context
def cov(ctx, sdate, edate, delta, deltaEigen, cov):
    ctx.invoke(handle, sdate = sdate, edate = edate, delta = delta, deltaEigen = deltaEigen, cov = cov)


@cov.command()
# main function
def handle(sdate, edate, delta, deltaEigen, cov):

    sdate = pd.Timestamp(sdate).strftime('%Y-%m-%d')
    edate = pd.Timestamp(edate).strftime('%Y-%m-%d')
    #
    startDate = pd.Timestamp(datetime.strptime(sdate, '%Y-%m-%d') - timedelta(days = delta+200)).strftime('%Y-%m-%d')
    startDateFlr = pd.Timestamp(datetime.strptime(sdate, '%Y-%m-%d') - timedelta(days = delta+200)).strftime('%Y-%m-%d')
   
    fr = factorReturn(startDate, edate)
    flr = loadFlr(startDateFlr, edate)
    fr.loc[:,'trade_date'] = fr['trade_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    flr.loc[:,'bf_date'] = flr['bf_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
 
    if len(fr[fr.trade_date >= sdate]) == 0 or len(flr[flr.bf_date >= sdate]) == 0:
        print('no data between sdate and edate! please check factor return table and fluctuation table. or you can change sdate/edate')
        exit()
    
    dates = list(set(fr.trade_date).intersection(set(flr.bf_date)))
    dates.sort()
    if len(dates) < delta+1:
        print('data is not enough for fitting time window. please check factor return table and fluctuation table. or you can change sdate/edate or time window')
        exit()

    datesHelper = [t for t in dates if t >= sdate]
    if len(dates) - len(datesHelper) < delta:
        sdate = dates[delta]
        startDateFlr = dates[0]
        startDate = dates[1]
    else:
        startDate = dates[len(dates)-len(datesHelper)-(delta-1)]
        startDateFlr = dates[len(dates) -len(datesHelper)-delta]

    fr = fr[fr.trade_date >= startDate]
    flr = flr[flr.bf_date >= startDateFlr]
    # flr used in the model  is not the real flr but the flr predicted by data delta days before
    dayMax = max(dates)
    flr = flr[flr['bf_date']<dayMax]
    datesFr = list(set(fr['trade_date']))
    datesFr.sort()
    datesFlr = list(set(flr['bf_date']))
    datesFlr.sort()
    flr['bf_date'] = flr['bf_date'].map(lambda x: datesFr[datesFlr.index(x)])
    
    # trade dates recorded in resid table and factor return table should be the same
    # trade dates recorded in residFlr table and flr table should be the same
    resid = regressionResid(startDate, edate)
    residFlr = loadResidFlr(startDateFlr, edate)
    resid.loc[:,'trade_date'] = resid['trade_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    residFlr.loc[:,'br_date'] = residFlr['br_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    # flr used in the model  is not the real flr but the flr predicted by data delta days before
    dayMax = max(dates)
    residFlr = residFlr[residFlr['br_date']<dayMax]
    datesFr = list(set(resid['trade_date']))
    datesFr.sort()
    datesFlr = list(set(residFlr['br_date']))
    datesFlr.sort()
    residFlr['br_date'] = residFlr['br_date'].map(lambda x: datesFr[datesFlr.index(x)])
    
    dates = set(resid['trade_date']).intersection(set(residFlr['br_date']).intersection(set(fr['trade_date']).intersection(set(flr['bf_date']))))
    dates =  list(dates)

    Exposure = factorExposure(dates, industryFactors)
    Exposure.loc[:,'trade_date'] = Exposure['trade_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
 
    dates = list(set(dates).intersection(set(Exposure['trade_date'])))
    dates.sort()

    factorNames = ['country'] + styleFactors + industryFactors
    
    fr = fr[fr['trade_date'].isin(dates)]
    flr = flr[flr['bf_date'].isin(dates)]
    fr = fr.set_index('trade_date').sort_index(axis = 0, ascending = True)
    flr = flr.set_index(['bf_date','bf_factor']).sort_index(level = 0, axis = 0, ascending = True).unstack()['bf_flr']
    fr = fr[factorNames]
    flr = flr[factorNames]

    # do some adjustment 
    # exponent weighed adjustment
    std, r, frStd, frR = exponentWeight(fr, sdate, halfLifeStd = 252, halfLifeR = 84, weight = True) 
    # newey-west adjustment
    sigma = neweyWest(std, r, frStd, frR, qStd = 5, qR = 2)
    # eigen adjustment ############# there is sth wrong with this ##############
    #sigma = eigen(sigma, M = 1000, alpha = 1.2, timeWindows = deltaEigen)
    # fluctuation ratio adjustment
    sigma = fluctuation(sigma, fr, flr, sdate)

    if len(sigma) == 0:
        print('covariance empty! please change sdate and edate!')
        exit()
    
    saveFactorCovariance(sigma, factorNames, list(fr[fr.index >= sdate].index))
    print('Factor return covariance saved! Check barra_factor_covariance to see more details.') 
    
    resid = resid[resid['trade_date'].isin(dates)]
    residFlr = residFlr[residFlr['br_date'].isin(dates)]
    resid = resid.set_index(['trade_date','stock_id']).sort_index(axis = 0, ascending = True)
    residFlr = residFlr.set_index(['br_date','br_stock_id']).sort_index(axis = 0, ascending = True)
    resid = resid.unstack()['resid']
    residFlr = residFlr.unstack()['br_flr']
    
    Exposure = Exposure[Exposure['trade_date'].isin(dates)]
    Exposure = Exposure.set_index('trade_date').sort_index(axis = 0, ascending = True)
    weight = Exposure[Exposure.index >= sdate] [['stock_id','weight']]
    exposure = Exposure[Exposure.index >= sdate] [['stock_id'] + factorNames] 

    Dates = [pd.Timestamp(date).strftime('%Y-%m-%d') for date in dates]
    Dates = [date for date in Dates if date >= sdate]
    Dates.sort()
    # do some adjustment on omiga
    # exponent weighed adjustment
    residExponent = exponentWeightOmiga(resid, sdate, halfLife = 84, weight = True)
    # newey-west adjustment
    omiga = neweyWestOmiga(residExponent, q = 5) ##### slow #######
    # structure model adjustment 
    omiga = structure(omiga, resid, residFlr, exposure, Dates, h = 252, E0 = 1.05)
    # bayes adjustment
    omiga = bayes(omiga, weight, Dates, groups = 10, q = 1)
    # fluctuation ratio adjustment
    omiga = fluctuationOmiga(omiga, resid, residFlr, Exposure[['stock_id','weight']], sdate, list(resid[resid.index >= sdate].index))

    saveStockVariance(omiga, Dates)
    print('stock variance saved! Check barra_stock_variance to see more details.')

    if cov != False:
        covarianceMatrixList = FactorCovariance(exposure, sigma, omiga, Dates)
        saveStockCovariance(covarianceMatrixList, Dates)
        print('Stock return covariance saved! Check barra_stock_covariance to see more details.')
