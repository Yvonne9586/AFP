# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
# os.chdir('C:/UCB/AFP')
#
# #data = pd.read_excel('GLOBAL.xlsx', 'LUATTRUU')
# xls = pd.ExcelFile('GLOBAL.xlsx')
# sheet_to_map = {}
# for sheet_name in xls.sheet_names:
#     sheet_to_map[sheet_name] = xls.parse(sheet_name).set_index('Date')
#
# df = pd.DataFrame(sheet_to_map['SPX INDEX']).rename(columns = {'Last Price': 'SPX INDEX'})
#
# for key in sheet_to_map.keys():
#     if key != 'MXEF':
#         df = df.merge(sheet_to_map[key], left_index = True, right_index = True).rename(columns = {sheet_to_map[key].columns[0] : key})
#
# df.to_csv('indexes.csv')

#this is my push for branch 1

#this is colins change

# process data from GFD
# allFiles = glob.glob("data/Tier1&2/*.csv")
# idx_dict = {}
# for file_ in allFiles:
#     idx = file_.split('/')[-1].split('.csv')[0]
#     df = pd.read_csv(file_, index_col='Date', skiprows=2)['Close']
#     df.index = pd.DatetimeIndex(df.index)
#     df = df[~df.index.duplicated(keep='first')]
#     idx_dict[idx] = df
# idx_df = pd.concat(idx_dict, axis=1)
# idx_df.to_csv("data/gfd.csv")
# idx_df.resample('M').last().to_csv("data/gfd_monthly.csv")
# returns = idx_df.resample('M').last().pct_change()

# Tier 3 data
allFiles = glob.glob("data/Tier3/*.csv")
idx_dict = {}
for file_ in allFiles:
    idx = file_.split('/')[-1].split('.csv')[0]
    df = pd.read_csv(file_, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep='first')]
    idx_dict[idx] = df
idx_df = pd.concat(idx_dict, axis=1)
idx_df.columns = ['BAB', 'CS', 'UMD_Large', 'UMD_Small']
idx_df.loc[:, ['BAB', 'UMD_Large', 'UMD_Small']] = (idx_df.loc[:, ['BAB', 'UMD_Large', 'UMD_Small']] + 1).cumprod()
tier1_df = pd.read_csv("data/gfd.csv", index_col='Date')
tier1_df.index = pd.DatetimeIndex(tier1_df.index)
combine_df = pd.merge(tier1_df, idx_df, left_index=True, right_index=True, how='outer')
combine_df.resample('M').last().to_csv("data/combined_dataset_new.csv")
print(combine_df.head().to_string())

# # Factor data
# allFiles = glob.glob("data/Others/*.csv")
# idx_dict = {}
# for file_ in allFiles:
#     idx = file_.split('/')[-1].split('.csv')[0]
#     df = pd.read_csv(file_, index_col=0)
#     df.index = pd.DatetimeIndex(df.index)
#     df = df[~df.index.duplicated(keep='first')]
#     idx_dict[idx] = df
# idx_df = pd.concat(idx_dict, axis=1)
# idx_df.columns = ['CPI', 'GDP', 'VIX', 'UMD', 'HML', 'SMB']
# idx_df.loc[:, ['CPI', 'GDP', 'VIX']] = idx_df.loc[:, ['CPI', 'GDP', 'VIX']].pct_change()
# idx_df.to_csv("data/factor_return.csv")
# print(idx_df.head().to_string())