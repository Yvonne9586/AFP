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
from scipy.spatial.distance import cdist


#adjusted sharpe ratio
def asp(ret):
    sr = ret.mean()/ret.std()*np.sqrt(12)
    s3 = skew(ret)
    s4 = kurtosis(ret)
    return sr*(1+ s3/6*sr  - (s4 - 3)/24*sr**2)

def ceq(ret, gamma):
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
    return mdd
    
def turnover(weights):
    sum = 0
    for i in range(weights.shape[1]-1):
        sum += np.abs(weights.iloc[:,i] - weights.iloc[:,i+1]).sum()
    return sum/weights.shape[1]*12

def SSPW(weights):
    return np.sum(np.sum(weights**2))/weights.shape[1]
    







