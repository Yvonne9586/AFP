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
os.chdir('C:/UCB/AFP')
from scipy.stats import skew,kurtosis

#data = pd.read_excel('GLOBAL.xlsx', 'LUATTRUU')
xls = pd.ExcelFile('GLOBAL2.xlsx')
sheet_to_map = {}
for sheet_name in xls.sheet_names:
    sheet_to_map[sheet_name] = xls.parse(sheet_name).set_index('Date')

df = pd.DataFrame(sheet_to_map['SPX INDEX']).rename(columns = {'Last Price': 'SPX INDEX'})

for key in sheet_to_map.keys():
    if (key != 'MXEF' and key!= 'SPX INDEX'):
        df = df.merge(sheet_to_map[key], left_index = True, right_index = True).rename(columns = {sheet_to_map[key].columns[0] : key})


###write index data to csv
df.to_csv('indexes.csv')    

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


def getClusterVar(cov, cItems):
    cov_ = cov.loc[cItems,cItems]
    w_ = getIVP(cov_).reshape(-1,1)
    cVar = np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getRecBipart(cov, sortIx):
    w = pd.Series(1, index = sortIx)
    cItems = [sortIx]
    while len(cItems)>0:
        cItems = [i[j:k] for i in cItems for j,k in ((0, int(len(i)/2)),(int(len(i)/2), len(i))) if len(i)>1]
        for i in range(0,len(cItems),2):
            cItems0 = cItems[i]
            cItems1 = cItems[i+1]
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha
            w[cItems1]*=1- alpha
    return w


#log return
ret_df = np.log(1+ df.pct_change())
ret_df = ret_df.sort_index()
ret_df = ret_df[(ret_df.index>=datetime.datetime(1982,1,1)) & (ret_df.index< datetime.datetime(2019,1,1))]
ret_df = ret_df.dropna(how = 'any')
ret_df = ret_df.drop(columns = ['NZDUSD'])
year_min = ret_df.index.year.min()
year_max = ret_df.index.year.max()

#one slump clustering
cov,corr = ret_df.cov(), ret_df.corr()
dist = ((1 - corr/2.))**.5
link = sch.linkage(dist, 'single')
fig = plt.figure(figsize = (25,10))
dn = dendrogram(link, labels = ret_df.columns )
plt.show()


#moving window clustering
oos = {}
w = {}
train_period = 300
test_period = 10
for i in range(0,2*(35*12 +11)):
    train = ret_df.iloc[i*test_period:(train_period + i*test_period) ,:]
    test = ret_df.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    cov, corr = train.cov(), train.corr()
    dist = ((1- corr/2.))**.5
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx= corr.index[sortIx].tolist()
    df0 = corr.loc[sortIx, sortIx]
    hrp = pd.DataFrame(getRecBipart(cov, sortIx)).T
    hrp = hrp/hrp.sum(axis = 1)[0]
    w[i] = hrp.T
    hrp.index= ['weight']
    test = pd.concat([test, hrp],join = 'inner')
    oos[i] = pd.DataFrame(np.array(test.iloc[-1,:])*test.iloc[:-1,:]).sum(axis = 1)
    
    
oos_test = pd.DataFrame(oos[0]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[0]).rename(columns = {0: ret_df.index[train_period + 1]})


for key in oos.keys():
    if key != 0:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
        weights = weights.merge(pd.DataFrame(w[i]).rename(columns = {0: ret_df.index[train_period + key*test_period + 1]}), left_index =  True, right_index = True)
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot((1+oos_test).cumprod())

#diagnostic
ret_df[(ret_df.index >= datetime.datetime(1986,1,1)) & (ret_df.index< datetime.datetime(1988,1,1))].plot()





#adjusted sharpe ratio
def asp(ret):
    sr = ret.mean()/ret.std()*np.sqrt(250)
    s3 = skew(ret)
    s4 = kurtosis(ret)
    return sr*(1+ s3/6*sr  - (s4 - 3)/24*sr**2)

def ceq(ret, gamma):
    rf = ret_df['US10Y'].mean()*250
    mu = ret.mean()*250
    sigma = ret.std()*np.sqrt(250)
    return (mu - rf) - gamma*0.5*sigma**2

 
def mdd(ret):
    ret = ret.add(1)
    ret = ret.apply(pd.to_numeric)
    dd = ret.divide(ret.cummax()).sub(1)
    # dd = r.sub(r.cummax())
    mdd = dd.min()
    end = dd.idxmin()
    start = ret.loc[:end].idxmax()
    return mdd, start, end
    
def turnover(weights):
    sum = 0
    for i in range(weights.shape[1]-1):
        sum += np.abs(weights.iloc[:,i] - weights.iloc[:,i+1]).sum()
    return sum/weights.shape[1]

def SSPW(weights):
    return np.sum(np.sum(weights**2))/weights.shape[1]
    







