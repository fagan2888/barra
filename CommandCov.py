
import pandas as pd
from sqlalchemy import *
from datetime import datetime, timedelta
import numpy as np
from db.config import *
from db.factors import styleFactors, industryFactors
from db import database
from db.operations import *
from CommandMatrixAdjust import *
from ipdb import set_trace
import click

# payback and resid is calculated in CommandCal.py
industryFactors.sort()

@click.group(invoke_without_command = True)
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime(2018,1,2)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime(2020,1,21)).strftime('%Y-%m-%d'), help = 'end date')
@click.pass_context
def cov(ctx, sdate, edate):
    ctx.invoke(handle, sdate = sdate, edate = edate)


@cov.command()
@click.option('--sdate', 'sdate', default = pd.Timestamp(datetime(2018,1,2)).strftime('%Y-%m-%d'), help = 'start date')
@click.option('--edate', 'edate', default = pd.Timestamp(datetime(2020,1,21)).strftime('%Y-%m-%d'), help = 'end date')
@click.pass_context
# main function
def handle(ctx, sdate, edate):

    weight = factorExposure(date, industryFactors, stocks)
    weight.sort_values(by = ['trade_date','stock_id'],ascending = True, inplace = True)
    weight.set_index(['trade_date','stock_id'], inplace = True)
    weight.sort_index(axis = 1 , inplace = True)
    w = np.matrix(weight)
    

    fr = factorReturn(sdate, edate)
    flr = loadFlr(sdate, edate)
    fr = np.matrix(fr).T
    flr = np.matrix(flr).T
    print(fr)
    set_trace()
    print(flr)
    set_trace()
    # fr and flr should be in the same shape!
    sigma = np.cov(fr)
    # do some adjustment 
    # all input should be in form of matrix
#    sigma = nothing(sigma)
    # exponent weighed adjustment
    sigma = exponentWeight(fr, halfLifeStd = 252, halfLifeR = 84)
    # newey-west adjustment
    sigma = neweyWest(sigma, fr, qStd = 5, qR = 2, halfLifeStd = 252, halfLifR = 84)
    # eigen adjustment
    sigma = eigen(sigma, fr, M = 1000, alpha = 1.2)
    # fluctuation ratio adjustment
    sigma = fluctuation(sigma, fr, flr)

    # do some adjustment on omiga
    resid = regressionResid(edate, edate, stocks) 
    residFlr = loadResidFlr(sdate, edate)
    # if do nothing
    omiga = np.diag(resid.apply(lambda x: x**2).mean())
    # all input should be in form of matrix
    resid = np.mat(resid).T
    residFlr = np.mat(residFlr).T
    # exponent weighed adjustment
#    omiga = exponentWeight(resid, halfLifeStd = 84, halfLifeR = 84)
    # newey-west adjustment
#    omiga = neweyWest(omiga, resid, qStd = 5, qR = 5, halfLifeStd = 84, halfLife = 84)
    # structure model adjustment
#    omiga = structure(omiga, r, exposure, omiga, h = 252, E0 = 1.05)
    # bayes adjustment
#    omiga = bayes(omiga, weight, groups = 10, q = 1)
    # fluctuation ratio adjustment
#    omiga = fluctuation(omiga, resid, residFlr)

    covarianceMatrix = FactorCovariance(w, sigma, omiga)
    
    saveFactorCovariance(sigma, factorNames, date)
    print('Factor return covariance saved! Check barra_factor_covariance to see more details.') 
    saveStockCovariance(covarianceMatrix, stockIds, date)  ##### slow #####
    print('Stock return covariance saved! Check barra_stock_covariance to see more details.')


if __name__ == '__main__':
    sdate = pd.Timestamp(datetime(2018,1,2)).strftime('%Y-%m-%d')
    edate = pd.Timestamp(datetime(2020,1,21)).strftime('%Y-%m-%d')
    handle(sdate, edate)
