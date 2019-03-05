# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:43:36 2019

@author: yangy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
os.chdir('C:/UCB/AFP/newAFP/AFP')
from scipy.stats import skew , kurtosis
import HRP
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist


#temp = {}
#for file in os.listdir():
#    temp[file] = pd.read_csv(file, skiprows = 2)[['Date','Close']]
#temp['Gold.csv'] = temp['Gold.csv'].iloc[2000:,]
#df = pd.DataFrame(temp['USEq.csv'])
#df = df.set_index(pd.to_datetime(df.Date), drop = True)
#del df['Date']
#df = df.rename(columns = {'Close':'USEq'})
#df = df.resample('M').last()

#for file in os.listdir():
#    if file != 'USEq.csv':
#        temp_df = pd.DataFrame(temp[file])
#        temp_df = temp_df.set_index(pd.to_datetime(temp_df.Date))
#        del temp_df['Date']
#        temp_df = temp_df.rename(columns = {'Close':file[:-4]})
#        temp_df = temp_df.resample('M').last()
#        df = df.merge(temp_df, how = 'inner', left_index = True, right_index = True)
        
gfd = pd.read_csv('data/gfd_monthly.csv')
gfd = gfd.set_index(pd.to_datetime(gfd['Date']))
gfd = gfd.iloc[:-1,:]
del gfd['Date']
gfd = gfd.pct_change()
del gfd['REITs']
gfd = gfd.dropna()

      
factor = pd.read_csv('data/factor_data.csv')
factor = factor.set_index(pd.to_datetime(factor['Date']))
del factor['Date']
factor = factor.resample('M').apply(lambda x: (1+x).cumprod().iloc[-1,:] - 1)

cs = pd.read_excel('data/CS Managed Futures Liquid Index Daily.xlsx')
cs = cs.set_index(pd.to_datetime(cs['Date'])).sort_index()
del cs['Date']
liquid = cs.resample('M').last()
liquid = liquid.pct_change()

ret_month = pd.merge(pd.merge(gfd, factor, left_index =True, right_index = True, how = 'inner'), liquid, left_index = True, right_index = True, how = 'inner')
del ret_month['USBondInt']
ret_month.to_csv('ret_month.csv')


ret_month = ret_month.rename(columns = {'Value (HML FF)':'Value'})
ret_month = ret_month.rename(columns = {'Momentum (UMD)':'Momentum'})
ret_month = ret_month.rename(columns = {'Size (SMB)':'Size'})


####read new data###
combined = pd.read_csv('data/combined_data_new.csv')
combined = combined.rename(columns = {'Unnamed: 0':'Date'})
combined = combined.set_index(pd.to_datetime(combined['Date']))
del combined['Date']
#change germanbond before 1923/11/30 to nan
combined['GermanBond10Y'][:1648] = np.nan
combined_ret = combined.pct_change()
combined_ret[['BAB','UMD_Large','UMD_Small']] = combined[['BAB','UMD_Large','UMD_Small']]
#return series check
combined_ret.plot(figsize = (15,10))
plt.xticks()
ret_month = combined_ret
##data available date
date = {}
for i in combined_ret.columns:
    date[i] = combined_ret[i].dropna().index[0]
first_availabe = pd.DataFrame.from_dict(date, orient = 'index').rename(columns = {0:'first available date'})
    

#################One slump clustering###############
cov,corr = ret_month.cov(), ret_month.corr()
dist = ((1 - corr)/2)**.5

link = sch.linkage(dist, 'single')
fig = plt.figure(figsize = (25,10))
dn = dendrogram(link, labels = ret_month.columns )
plt.show()



#################cov hrp############################

oos = {}
w = {}
train_period = 24
test_period = 12
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[i*test_period:(train_period + i*test_period) ,:]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    train = train.dropna(axis = 1, how = 'any')
    train = train.iloc[:,(train != 0).any().values]
    cov, corr = train.cov(), train.corr()
    mean = train.mean()
    if train.shape[1]>1:
        dist = ((1- corr)/2)**.5
        link = sch.linkage(dist, 'single')
        sortIx = HRP.getQuasiDiag(link)
        sortIx= corr.index[sortIx].tolist()
        df0 = corr.loc[sortIx, sortIx]
        hrp = pd.DataFrame(HRP.getRecBipart(cov, sortIx, mean)).T
        #hrp = hrp/hrp.sum(axis = 1)[0]
        hrp[np.abs(hrp)>1] = 1
        w[i] = hrp.T
        hrp.index= ['weight']
        test = pd.concat([test, hrp])
        oos[i] = pd.DataFrame(np.array(test.iloc[-1,:])*test.iloc[:-1,:]).sum(axis = 1)
    
    
oos_test = pd.DataFrame(oos[list(oos.keys())[0]]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[list(w.keys())[0]]).rename(columns = {0: ret_month.index[train_period + 1]}).T


for key in oos.keys():
    if key != list(oos.keys())[0]:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
for key in w.keys():
    if key != list(w.keys())[0]:    
        weights = pd.concat([weights,w[key].T]).rename(index = {0: ret_month.index[train_period + key*test_period + 1]})
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot(1+oos_test.cumsum())
plt.xticks(rotation = 60)

weights.plot(figsize = (15,10),colormap='tab20')

###############one slump dd clustering#####################
drawdown = -HRP.mdd(ret_month)
dist = np.zeros((drawdown.shape[0],drawdown.shape[0]))
for h in range(drawdown.shape[0]):
    for k in range(drawdown.shape[0]):
        dist[h,k] = np.abs(drawdown.iloc[h] - drawdown.iloc[k])
dist = pd.DataFrame(dist, columns = drawdown.index, index = drawdown.index)     
link = sch.linkage(dist, 'single')
fig = plt.figure(figsize = (25,10))
dn = dendrogram(link, labels = ret_month.columns )
plt.show()




###############drawdown hrp###############################
oos = {}
w = {}
train_period = 12*10
test_period = 12
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[:(train_period + i*test_period) ,:]
    train = train.dropna(axis = 1, how = 'all')
    #train = train.iloc[:,(train != 0).any().values]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    if train.shape[1]>1:
        cov = train.cov()
        corr = train.corr()
        drawdown = -HRP.mdd(train)
        dist = np.zeros((drawdown.shape[0],drawdown.shape[0]))
        for h in range(drawdown.shape[0]):
            for k in range(drawdown.shape[0]):
                dist[h,k] = np.abs(drawdown.iloc[h] - drawdown.iloc[k])
        dist = pd.DataFrame(dist, columns = drawdown.index, index = drawdown.index)     
        link = sch.linkage(dist, 'single')
        sortIx = HRP.getQuasiDiag(link)
        sortIx= corr.index[sortIx].tolist()
        df0 = corr.loc[sortIx, sortIx]
        hrp = pd.DataFrame(HRP.getRecBipart(cov, sortIx, mean)).T
        hrp[np.abs(hrp)>1] = 1
        hrp = hrp/hrp.sum(axis = 1)[0]
        w[i] = hrp.T
        hrp.index= ['weight']
        test = pd.concat([test, hrp],join = 'inner')
        oos[i] = pd.DataFrame(np.array(test.iloc[-1,:])*test.iloc[:-1,:]).sum(axis = 1)
    
    
oos_test = pd.DataFrame(oos[list(oos.keys())[0]]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[list(w.keys())[0]]).rename(columns = {0: ret_month.index[train_period + 1]}).T


for key in oos.keys():
    if key != list(oos.keys())[0]:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
for key in w.keys():
    if key != list(w.keys())[0]:      
        weights = pd.concat([weights,w[key].T]).rename(index = {0: ret_month.index[train_period + key*test_period + 1]})
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot(oos_test.cumsum())
plt.xticks(rotation = 60)
#plot weights
weights = weights.fillna(method = 'ffill')
weights.plot(figsize = (25,10),colormap='tab20')

######Equity Betas#######
##time varying beta###
beta = {}
r2 = {}
window_length = 240
roll_beta = ret_month[ret_month.index>datetime.datetime(1871,3,1)]
for i in range(0,roll_beta.shape[0] - window_length - 120,120):
    data = roll_beta.iloc[i:(i+window_length),:]
    data = data.dropna(axis = 1, how = 'all')
    if data.shape[1]>3:
        beta[roll_beta.index[i]], r2[roll_beta.index[i]] = window_beta4(data)

betas = pd.DataFrame.from_dict(beta).T
betas.plot(figsize = (15,10), colormap='tab20')
pd.DataFrame.from_dict(r2, orient = 'index').plot()

        
ret_month[(ret_month.index>datetime.datetime(1913,1,1))&(ret_month.index<datetime.datetime(1915,1,1))].plot()


###equity beta hrp####

oos = {}
w = {}
train_period = 12*10
test_period = 12
ret_month = ret_month[ret_month.index>datetime.datetime(1871,3,1)]
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[:(train_period + i*test_period) ,:]
    train = train.dropna(axis = 1, how = 'all')
    #train = train.iloc[:,(train != 0).any().values]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    if train.shape[1]>1:
        cov = train.cov()
        corr = train.corr()
        beta = beta4_standardized(train)
        beta = beta.dropna()
        dist = np.zeros((beta.shape[0],beta.shape[0]))
        for h in range(beta.shape[0]):
            for k in range(beta.shape[0]):
                dist[h,k] = np.abs(beta.iloc[h] - beta.iloc[k])
        dist = pd.DataFrame(dist, columns = beta.index, index = beta.index)     
        link = sch.linkage(dist, 'single')
        sortIx = HRP.getQuasiDiag(link)
        sortIx= corr.index[sortIx].tolist()
        df0 = corr.loc[sortIx, sortIx]
        hrp = pd.DataFrame(HRP.getRecBipart(cov, sortIx, mean)).T
        hrp[np.abs(hrp)>1] = 1
        hrp = hrp/hrp.sum(axis = 1)[0]
        w[i] = hrp.T
        hrp.index= ['weight']
        test = pd.concat([test, hrp],join = 'inner')
        oos[i] = pd.DataFrame(np.array(test.iloc[-1,:])*test.iloc[:-1,:]).sum(axis = 1)
    
    
oos_test = pd.DataFrame(oos[list(oos.keys())[0]]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[list(w.keys())[0]]).rename(columns = {0: ret_month.index[train_period + 1]}).T


for key in oos.keys():
    if key != list(oos.keys())[0]:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
for key in w.keys():
    if key != list(w.keys())[0]:      
        weights = pd.concat([weights,w[key].T]).rename(index = {0: ret_month.index[train_period + key*test_period + 1]})
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot(oos_test.cumsum())
plt.xticks(rotation = 60)
#plot weights
weights = weights.fillna(method = 'ffill')
weights.plot(figsize = (25,10),colormap='tab20')






cov_weights = pd.read_csv('data/cov_weights.csv')
cov_weights = cov_weights.set_index(cov_weights['Unnamed: 0'])
del cov_weights['Unnamed: 0']
cov_weights.index.name = 'Date'
dd_weights = pd.read_csv('data/dd_weights.csv')
dd_weights = dd_weights.set_index(dd_weights['Unnamed: 0'])
del dd_weights['Unnamed: 0']
dd_weights.index.name = 'Date'

weights = pd.concat([cov_weights, dd_weights], axis = 0, keys = ['cov','dd'], names = ['method','Date'])
ax = weights.loc['cov'].plot(figsize = (20,8))
ax.set_xticklabels(weights.loc['cov'].index, rotation = 60)

weights.loc['dd'].plot(figsize = (20,8))
ax = weights.loc['dd'].plot(figsize = (20,8))
ax.set_xticklabels(weights.loc['dd'].index, rotation = 60)


