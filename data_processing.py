# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/UCB/AFP')

#data = pd.read_excel('GLOBAL.xlsx', 'LUATTRUU')
xls = pd.ExcelFile('GLOBAL.xlsx')
sheet_to_map = {}
for sheet_name in xls.sheet_names:
    sheet_to_map[sheet_name] = xls.parse(sheet_name).set_index('Date')

df = pd.DataFrame(sheet_to_map['SPX INDEX']).rename(columns = {'Last Price': 'SPX INDEX'})

for key in sheet_to_map.keys():
    if key != 'MXEF':
        df = df.merge(sheet_to_map[key], left_index = True, right_index = True).rename(columns = {sheet_to_map[key].columns[0] : key})

df.to_csv('indexes.csv')    