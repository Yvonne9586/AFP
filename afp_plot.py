# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:21:40 2019

@author: colin
"""
import pandas as pd
import pandas_datareader.data as web
#import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
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
#    USEq_df["last_month_USEq_return"] = USEq_df["USEq"].shift(-1)
#    USEq_df["US_Crisis"] = np.where((USEq_df["last_month_USEq_return"]*12<drawdown)
#                                     & (USEq_df["USEq"]*12<drawdown), 1, 0)
    USEq_df["US_Crisis"] = np.where((USEq_df["USEq"]*12<drawdown), 1, 0)
    USEq_df["US_Crisis_mean"] = USEq_df["US_Crisis"].rolling(3).mean()
    return USEq_df

def crisis_period_plot(tier1, total_return, run_env_name, plot_flag=2, drawdown=-0.2):
    """plot cumulative return with crisis period shaded in grey"""
    USEq_df = set_crisis_periods(tier1, drawdown)
    begin_date = '1999-02-01'
    end_date = '2018-12-31'
    _col = ["6040",
            "all-weather (star)",
            "risk_parity (tier 3)",
            "hrp (tier 3)",
            "hrp_dd (tier 3)",
            "hrp_beta (tier 3)",
            "hrp_val (tier 3)"
            ]
    if plot_flag == 2:
        begin_date = '1969-02-01'
        end_date = '2018-12-31'
        _col = ["6040",
                "all-weather (original)",
                "risk_parity (tier 2)",
                "hrp (tier 2)",
                "hrp_dd (tier 2)",
                "hrp_beta (tier 2)",
                "hrp_val (tier 2)",
                "hrp_strc (tier 2)"
                ]
    _total_return = total_return.loc[begin_date:end_date, _col].dropna()
    _total_return.apply(lambda x: x/x[0]).plot(grid=True, title='Cumulative Return', figsize=[12, 8])

    recs = ((USEq_df.query('US_Crisis_mean>=0.4')))
    _year = list(map(str,range(_total_return.index[0].year, _total_return.index[-1].year+1)))
    for year in _year:
        recs_2k = recs.loc[year]
        if recs_2k.shape[0] > 0:
            recs2k_bgn = recs_2k.index[0]
            recs2k_end = recs_2k.index[-1]        
            plt.axvspan(recs2k_bgn, recs2k_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
    plt.savefig("pic/total_return_%s.pdf" % (run_env_name+" tier " + str(plot_flag)))
    return None