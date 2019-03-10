# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:51:36 2019

@author: colin
"""
import scipy.cluster.hierarchy as sch, random, numpy as np, pandas as pd
from main import calc_final_results
from datetime import date
from dateutil.relativedelta import relativedelta


def generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    """Time series of correlated variables"""
    #1) generate random uncorrelated data
    # sigma0 is the arbitrary std of 10%
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    #2) create correlation between the variables
    cols = [random.randint(0, size0-1) for i in range(size1)]
    y = x[:,cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs,len(cols)))
    x = np.append(x, y, axis=1)
    #3) add common random shock
    point = np.random.randint(sLength, nObs-1, size=2)  # row num in the future year
    x[np.ix_(point,[cols[0], size0])] = np.array([[-.5,-.5], [2,2]])
    #4) add specific random shock
    point = np.random.randint(sLength, nObs-1, size=2)
    x[point, cols[-1]] = np.array([-.5, 2])
    x_df = pd.DataFrame(x)
    x_df["date"] = [date.today()+ relativedelta(months=i) for i in range(x_df.shape[0])]
    x_df.set_index("date", inplace=True)
    del x_df.index.name
    x_df.index = pd.to_datetime(x_df.index)
    return x_df, cols


def hrpMC(methods, numIters=100, nObs=360, size0=5, size1=5, mu0=0, sigma0=0.01,\
          sigma1F=.25, sLength=60, rebal=1):
    """Monte Carlo experiment on HRP"""
    # default: daily return data;
    #          use one year historical (sLength=260) to come up with weight
    #          then test the performance in the upcoming one month
    #          then rebalance (rebal=22)
    #          see the one year performance in the future (nObs=260 + sLength = 520)
    #          10 assets (size0=5 + size1=5)          
    numIter = 0
    total_return = {i: pd.DataFrame() for i in methods}
    results_metrics = {i: pd.DataFrame() for i in methods}
    while numIter < numIters:
        print(numIter)
        x, cols = generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
        returns_df = pd.DataFrame(x)
        for method in methods:
            print(numIter, method)
            method_name = method + " " + "MC_Iter_" + str(numIter)
            total_return[method], results_metrics[method] =\
                calc_final_results(total_return[method], results_metrics[method],
                                   returns_df=returns_df,
                                   method=method_name)
        numIter += 1
    return total_return, results_metrics