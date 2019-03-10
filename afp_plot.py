# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:21:40 2019

@author: colin
"""
import pandas as pd
import pandas_datareader.data as web

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))

import missingno as msno
p=print

def set_crisis_periods(USEq_df, drawdown):
    """define crisis period based on US equity's return
    periods with an annualized return of <-drawdown over a
    one-month-period in the U.S. equity market"""
    USEq_df["last_month_USEq_return"] = USEq_df["USEq"].shift(-1)
    USEq_df["US_Crisis"] = np.where((USEq_df["last_month_USEq_return"]*12<drawdown)
                                     & (USEq_df["USEq"]*12<drawdown), 1, 0)
    return USEq_df

def crisis_period_plot(tier1, total_return, drawdown=-0.2):
    """plot cumulative return with crisis period shaded in grey"""
    # recessions are marked as 1 in the data
    USEq_df = set_crisis_periods(tier1, drawdown)
    recs = USEq_df.query('US_Crisis==1')

    # Select the two recessions over the time period
    recs_2k = recs.loc['2001']
    recs_2k8 = recs.loc['2008':'2011']
    
    # now we can grab the indices for the start
    # and end of each recession
    recs2k_bgn = recs_2k.index[0]
    recs2k_end = recs_2k.index[-1]

    recs2k8_bgn = recs_2k8.index[0]
    recs2k8_end = recs_2k8.index[-1]

    # Let's plot SPX and VIX cumulative returns with recession overlay
    plot_cols = ['6040', 'all-weather']

    # 2 axes for 2 subplots
#    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)
#    total_return[plot_cols].plot(subplots=True, ax=axes)
#    for ax in axes:
#        ax.axvspan(recs2k_bgn, recs2k_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
#        ax.axvspan(recs2k8_bgn, recs2k8_end,  color=sns.xkcd_rgb['grey'], alpha=0.5)
#    fig, ax = plt.plot(figsize=(10,7))
    total_return[plot_cols].plot(figsize=(10,7))
    plt.axvspan(recs2k_bgn, recs2k_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
    plt.axvspan(recs2k8_bgn, recs2k8_end,  color=sns.xkcd_rgb['grey'], alpha=0.5)
    return None        



