# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
#os.chdir('C:/UCB/AFP/data')
from scipy.stats import skew , kurtosis
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
import statsmodels.api as sm

#############################################

#tree clustering 
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram


def getQuasiDiag(link):
    # sort clustered items by distance

    link = link.astype(int)
    sortIx = pd.Series([link[-1,0],link[-1,1]])
    numItems = link[-1,3]
    while sortIx.max()>=(numItems):
        sortIx.index = range(0,sortIx.shape[0]*2,2)
        df0 = sortIx[sortIx>=(numItems)]
        i = df0.index; j = df0.values - numItems
        sortIx[i] = link[j,0]
        df0 = pd.Series(link[j,1], index = i+1)
        sortIx = sortIx.append(df0)
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()


def getIVP(cov,**kargs):
    ivp = 1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp


def getClusterVar(cov, cItems, mean):
    cov_ = cov.loc[cItems,cItems]
    mean_ = mean.loc[cItems]
    w_ = getIVP(cov_).reshape(-1,1)
    #cVar = np.dot(np.dot(w_.T,cov_),w_)[0,0] * np.dot(w_.T,mean_)[0] 
    cVar = np.sqrt(np.dot(np.dot(w_.T,cov_),w_)[0,0])
    return cVar


def getRecBipart(cov, sortIx, mean):
    w = pd.Series(1, index = sortIx)
    cItems = [sortIx]
    while len(cItems)>0:
        cItems = [i[j:k] for i in cItems for j,k in ((0, int(len(i)/2)),(int(len(i)/2), len(i))) if len(i)>1]
        for i in range(0,len(cItems),2):
            cItems0 = cItems[i]
            cItems1 = cItems[i+1]
            cVar0 = getClusterVar(cov, cItems0, mean)
            cVar1 = getClusterVar(cov, cItems1, mean)
            alpha = 1 - cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha
            w[cItems1]*=1- alpha
    return w




#moving window clustering on drawdown (expanding window)


#adjusted sharpe ratio
def asp(ret):
    sr = ret.mean()/ret.std()*np.sqrt(12)
    s3 = skew(ret)
    s4 = kurtosis(ret)
    return sr*(1+ s3/6*sr  - (s4 - 3)/24*sr**2)

def ceq(ret_df,ret, gamma):
    rf = 0.03
    mu = ret.mean()*12
    sigma = ret.std()*np.sqrt(12)
    return (mu - rf) - gamma*0.5*sigma**2

 
def mdd(ret):
    cum_ret = pd.DataFrame(np.exp(ret.cumsum()))
    cum_ret = cum_ret.apply(pd.to_numeric)
    dd = cum_ret.divide(cum_ret.cummax()).sub(1)
    # dd = r.sub(r.cummax())
    mdd = dd.min()
    #end = dd.idxmin()
    #start = ret.loc[:end].idxmax()
    return mdd[0]
    
def turnover(weights):
    sum = 0
    for i in range(weights.shape[1]-1):
        sum += np.abs(weights.iloc[:,i] - weights.iloc[:,i+1]).sum()
    return sum/weights.shape[1]*12

def SSPW(weights):
    return np.sum(np.sum(weights**2))/weights.shape[1]
    
def beta_convexity(asset, useq):
    useq_pos = [0 if useq.iloc[i]<0 else useq.iloc[i] for i in range(useq.shape[0])]
    useq_pos2 = [0 if useq.iloc[i]<0 else useq.iloc[i]**2 for i in range(useq.shape[0])]
    useq_neg = [0 if useq.iloc[i]>0 else useq.iloc[i] for i in range(useq.shape[0])]
    useq_neg2 = [0 if useq.iloc[i]>0 else useq.iloc[i]**2 for i in range(useq.shape[0])]

    temp = pd.DataFrame({pd.DataFrame(asset).columns[0]: asset, 'USEq_pos':useq_pos, 'USEq_neg':useq_neg, 'USEq_pos2': useq_pos2, 
                         'USEq_neg2':useq_neg2}, index = asset.index)
    temp = temp.dropna()
    reg = LinearRegression().fit(temp.iloc[:,1:],temp.iloc[:,0])
    r2 = reg.score(temp.iloc[:,1:],temp.iloc[:,0])
    
    return reg.coef_,r2


def beta4_standardized(train):
    betas = {}
    useq = train['USEq']
    useq_pos = [0 if useq.iloc[i]<0 else useq.iloc[i] for i in range(useq.shape[0])]
    useq_pos2 = [0 if useq.iloc[i]<0 else useq.iloc[i]**2 for i in range(useq.shape[0])]
    useq_neg = [0 if useq.iloc[i]>0 else useq.iloc[i] for i in range(useq.shape[0])]
    useq_neg2 = [0 if useq.iloc[i]>0 else useq.iloc[i]**2 for i in range(useq.shape[0])]
    for asset in train.columns:
        if asset != 'USEq':
            temp = pd.DataFrame({asset: np.array(train[asset]), 'USEq_pos':useq_pos, 'USEq_neg':useq_neg, 'USEq_pos2': useq_pos2, 
                         'USEq_neg2':useq_neg2}, index = train.index)
        temp = temp.dropna()
        reg = sm.OLS(temp.iloc[:,0],temp.iloc[:,1:]).fit()
        r = np.zeros_like(reg.params)
        r[3] = 1
        T_test = reg.t_test(r)
        betas[asset] = reg.params[3]/T_test.sd[0][0]
    return pd.DataFrame.from_dict(betas, orient = 'index').rename(columns = {0:train.index[-1]})
    
    



def window_beta4(ret_month):
    beta = {}
    r2 = {}
    for asset in ret_month.columns:
        if asset!='USEq':
            beta[asset] = beta_convexity(ret_month[asset], ret_month['USEq'])[0][3]
            r2 = beta_convexity(ret_month[asset], ret_month['USEq'])[1]
    return beta,r2

import statsmodels.api as sm
def structure_break(price_month):
    SADF = {}
    temp = []
    log_price = np.log(price_month)
    tau = 60
    L = 4
    log_price_dff = log_price.diff()
    t =log_price.shape[0] - 1
    for asset in price_month.columns:
        for t0 in range(L, t - tau, 12):
            reg_data = pd.DataFrame({
                    'dy_t':np.array(log_price[asset].iloc[(t0+1):t]),
                    'alpha':1, 
                    'y_t-1':np.array(log_price[asset].iloc[t0:(t-1)]),
                    'dy_t-1':np.array(log_price_dff[asset].iloc[t0:(t-1)]),
                    'dy_t-2': np.array(log_price_dff[asset].iloc[(t0-1):(t-2)]),
                    'dy_t-3': np.array(log_price_dff[asset].iloc[(t0-2):(t-3)]),
                    'dy_t-4': np.array(log_price_dff[asset].iloc[(t0-3):(t-4)]),
                    'dy_t-5': np.array(log_price_dff[asset].iloc[(t0-4):(t-5)])}, index = log_price_dff.iloc[t0+1:t,].index).dropna()
            reg = sm.OLS(reg_data['dy_t'],reg_data[['alpha','y_t-1','dy_t-1','dy_t-2','dy_t-3','dy_t-4','dy_t-5']]).fit()
            r = np.zeros_like(reg.params)
            r[1] = 1
            T_test = reg.t_test(r)
            temp.append(reg.params[1]/T_test.sd[0][0])
        SADF[asset] = np.max(temp)
    return SADF
        




