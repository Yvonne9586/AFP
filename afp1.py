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

#################One slump clustering###############
cov,corr = ret_month.cov(), ret_month.corr()
dist = ((1 - corr/2.))**.5
link = sch.linkage(dist, 'single')
fig = plt.figure(figsize = (25,10))
dn = dendrogram(link, labels = ret_month.columns )
plt.show()



#################cov hrp############################

oos = {}
w = {}
train_period = 24
test_period = 5
for i in range(0, int((ret_month.shape[0] - train_period)/test_period)):
    train = ret_month.iloc[i*test_period:(train_period + i*test_period) ,:]
    test = ret_month.iloc[(train_period + i*test_period +1):(train_period + i*test_period +1 + test_period)  ,:]
    cov, corr = train.cov(), train.corr()
    mean = train.mean()
    dist = ((1- corr/2.))**.5
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
plt.title('HRP Cov')


#######
