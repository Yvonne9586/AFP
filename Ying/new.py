# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:39:36 2019

@author: yangy
"""

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
os.chdir('C:/UCB/AFP/AFP')
from scipy.stats import skew , kurtosis
import HRP
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from sklearn.linear_model import LinearRegression

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

#link = sch.linkage(dist, 'single')
#fig = plt.figure(figsize = (25,10))
#dn = dendrogram(link, labels = ret_month.columns )
#plt.show()



#################cov hrp############################
oos = {}
w = {}
train_period = 24
test_period = 3
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[i*test_period:(train_period + i*test_period) ,:]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    cov, corr = train.cov(), train.corr()
    mean = train.mean()
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
    test = pd.concat([test, hrp],join = 'inner')
    oos[i] = pd.DataFrame(np.array(test.iloc[-1,:])*test.iloc[:-1,:]).sum(axis = 1)
    
    
oos_test = pd.DataFrame(oos[0]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[0]).rename(columns = {0: ret_month.index[train_period + 1]})


for key in oos.keys():
    if key != 0:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
        weights = weights.merge(pd.DataFrame(w[key]).rename(columns = {0: ret_month.index[train_period + key*test_period + 1]}), left_index =  True, right_index = True)
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot(np.exp(oos_test.cumsum()))
plt.xticks(rotation = 60)

#plot weights
weights.T.plot(figsize = (15,10))

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
train_period = 12*2
test_period = 3
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[:(train_period + i*test_period) ,:]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    cov = train.cov()
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
    
    
oos_test = pd.DataFrame(oos[0]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[0]).rename(columns = {0: ret_month.index[train_period + 1]})


for key in oos.keys():
    if key != 0:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
        weights = weights.merge(pd.DataFrame(w[key]).rename(columns = {0: ret_month.index[train_period + key*test_period + 1]}), left_index =  True, right_index = True)
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot(np.exp(oos_test.cumsum()))
plt.xticks(rotation = 60)

weights.T.plot(figsize = (15,10))

##############################one slump Downside beta###############################
def downside_beta(asset, useq):
    useq_neg = pd.DataFrame({'USEq_neg':[0 if useq[i]>0 else useq[i] for i in range(useq.shape[0])]}, index = useq.index)
    beta = []
    for i in asset.columns:      
        data = pd.concat([asset[i] ,useq_neg],join = 'inner', axis = 1)
        data = data.dropna()
        reg = LinearRegression().fit(data.iloc[:,1:],data.iloc[:,0])
        beta.append(reg.coef_[0])
    return beta

beta = downside_beta(ret_month, ret_month['USEq'])
dist = np.zeros((len(beta),len(beta)))
for h in range(len(beta)):
    for k in range(len(beta)):
        dist[h,k] = np.abs(beta[h] - beta[k])
dist = pd.DataFrame(dist, columns = train.columns, index = train.columns) 
link = sch.linkage(dist, 'single')
fig = plt.figure(figsize = (25,10))
dn = dendrogram(link, labels = ret_month.columns )
plt.show()


###########################hrp downside beta#######################################
oos = {}
w = {}
train_period = 24
test_period = 3
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[i*test_period:(train_period + i*test_period) ,:]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    cov, corr = train.cov(), train.corr()
    mean = train.mean()
    beta = downside_beta(train, train['USEq'])
    dist = np.zeros((len(beta),len(beta)))
    for h in range(len(beta)):
        for k in range(len(beta)):
            dist[h,k] = np.abs(beta[h] - beta[k])
    dist = pd.DataFrame(dist, columns = train.columns, index = train.columns) 
    link = sch.linkage(dist, 'single')
    sortIx = HRP.getQuasiDiag(link)
    sortIx= corr.index[sortIx].tolist()
    df0 = corr.loc[sortIx, sortIx]
    hrp = pd.DataFrame(HRP.getRecBipart(cov, sortIx, mean)).T
    #hrp = hrp/hrp.sum(axis = 1)[0]
    hrp[np.abs(hrp)>1] = 1
    w[i] = hrp.T
    hrp.index= ['weight']
    test = pd.concat([test, hrp],join = 'inner')
    oos[i] = pd.DataFrame(np.array(test.iloc[-1,:])*test.iloc[:-1,:]).sum(axis = 1)
    
    
oos_test = pd.DataFrame(oos[0]).rename(columns = {0:'ret'})
weights = pd.DataFrame(w[0]).rename(columns = {0: ret_month.index[train_period + 1]})


for key in oos.keys():
    if key != 0:
        oos_test = pd.concat([oos_test,pd.DataFrame(oos[key]).rename(columns = {0:'ret'})])
        weights = weights.merge(pd.DataFrame(w[key]).rename(columns = {0: ret_month.index[train_period + key*test_period + 1]}), left_index =  True, right_index = True)
#plot out of sample performance
oos_test = oos_test.sort_index()
plt.plot(np.exp(oos_test.cumsum()))
plt.xticks(rotation = 60)

weights.T.plot(figsize = (15,10))

######################analysis##################################
dd = pd.read_csv('data/dd_backtest.csv')
cov = pd.read_csv('data/cov_backtest.csv')
beta = pd.read_csv('data/beta_backtest.csv')

def evaluation(backtest, name):
    return pd.DataFrame({'ASR':HRP.asp(backtest),'CEQ': HRP.ceq(backtest,1),
                         'MDD': HRP.mdd(backtest)}, index = [name])

evaluation(dd['ret'], 'dd')
evaluation(cov['ret'], 'cov')
evaluation(beta['ret'], 'beta')






