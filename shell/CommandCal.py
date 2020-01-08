
import numpy as np
import pandas as pd
from sqlalchemy import *
from config import *
from db import database
from datetime import datetime, timedelta, date
import statsmodels.api as sm
from ipdb import set_trace

# work out some basic data for barra
class calculate(object):

    def __init__(self, sdate, edate):
        self.sdate = sdate
        self.edate = edate

    def handle(self):

        sdate = self.sdate
        edate = self.edate
        
        # create dataframe of factor returns
        styleFactors = [
            'volatility',
            'dividend_yield',
            'quality',
            'momentum',
            'short_term_reverse',
            'value',
            'linear_size',
            'nonlinear_size',
            'growth',
            'liquidity',
            'sentiment',
        ]
        styleFactors.sort()
        industryFactors = [
            'industry_6710100000',
            'industry_6715100000',
            'industry_6720100000',
            'industry_6720200000',
            'industry_6720300000',
            'industry_6725100000',
            'industry_6725200000',
            'industry_6725300000',
            'industry_6725400000',
            'industry_6725500000',
            'industry_6730100000',
            'industry_6730200000',
            'industry_6730300000',
            'industry_6735100000',
            'industry_6735200000',
            'industry_6740100000',
            'industry_6740200000',
            'industry_6740400000',
            'industry_6745100000',
            'industry_6745200000',
            'industry_6745300000',
            'industry_6750200000',
            'industry_6755100000',
            'industry_6760100000',
        ]
        industryFactors.sort()
        
        # load factor exposures of every stocks
        db = create_engine(uris['multi_factor'])
        sql = "select * from `factor_exposure_barra` where trade_date >= '" + sdate + "' and trade_date <='" + edate + "'"
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
        sql = sql.where(t.c.S_DQ_TRADESTATUS != '停牌').where(t.c.S_DQ_TRADESTATUS != '待核查')
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
        for stock in stocks:
            dfTmp = dfAdjCloseG.get_group(stock).copy()
            dfTmp.sort_values(by = 'TRADE_DT', ascending = True, inplace = True)
            dfTmp.reset_index(inplace = True, drop = True)
            pct_change = dfTmp['S_DQ_ADJCLOSE'].pct_change()
            dfTmp.insert(0, 'pct_change', pct_change)
            dfTmp = dfTmp.fillna(0)
            dfAdjClose = pd.concat([dfAdjClose, dfTmp] , axis = 0, sort = True)
        dfAdjClose.drop_duplicates(['TRADE_DT','S_DQ_ADJCLOSE'], inplace = True)
        dfAdjCloseG = dfAdjClose.groupby('TRADE_DT')        

        # main part
        dfResid = pd.DataFrame(columns = ['trade_date','stock_id','resid'])
        dfParams = pd.DataFrame(columns = ['trade_date'] + ['country'] + styleFactors + industryFactors)
        dfParams.set_index('trade_date', inplace = True)
        # rn = fc + Sigma(Xi*fi) + Sigma(Xs*fs) + un  Sigma(w*fi) = 0  un is resid
        for date, exposure in dfExposureG:
            dateWind = pd.Timestamp(date).strftime('%Y%m%d')
            dfAdjClose = dfAdjCloseG.get_group(dateWind).copy()
            dfAdjClose = dfAdjClose.fillna(0)
            exposure = exposure[exposure['stock_id'].isin(dfAdjClose['S_INFO_WINDCODE'])]
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
            Q = np.hstack([[0],w,np.zeros(len(styleFactors))])
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
            dfParams = pd.concat([dfParams, dfP],axis = 0)
            
            dfR = pd.DataFrame(columns = ['trade_date','stock_id','resid'])
            dfR['stock_id'] = exposure['stock_id'] 
            dfR['resid'] = resid
            dfR['trade_date'] = date
            dfResid = pd.concat([dfResid, dfR], axis = 0)
        
        dfParams.sort_index(axis = 1, inplace = True)
        # connect to database and update factor returns
        db = create_engine(uris['multi_factor'])
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
        dfBase = pd.read_sql(sql, db)
        dfBase.set_index(['trade_date','stock_id'], inplace = True)

        database.batch(db,t,dfResid,dfBase,timestamp = False)
        print('regression reside updated!')

if __name__ == '__main__':
    edate = datetime.today()
    edate = date(2019,12,31)
    sdate = pd.Timestamp(edate-timedelta(days = 5)).strftime('%Y-%m-%d')
    edate = pd.Timestamp(edate).strftime('%Y-%m-%d')
    cal = calculate(sdate, edate)
    cal.handle()
