# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:14:09 2018

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_name = 'ft_kurt.xlsx'
xl_workbook = pd.ExcelFile(file_name)  # Load the excel workbook
df = xl_workbook.parse("Plot Values_FFT")  # Parse the sheet into a dataframe
delf = np.array(df['Frequency'].tolist())
g    = np.array(df['100lppi_L_good'].tolist())  # Cast the desired column into a python list
b12  = np.array(df['100lppi_L_geardefect_12T'].tolist())
b60  = np.array(df['100lppi_L_brokengear60T'].tolist())

def find_max(col,left,right,maxnum):
    max_abs = col[left:right][(np.argsort(col[left:right])[-maxnum:])]
    return max_abs

g_c = find_max(g,38,44,1)
g_d = find_max(g,174,293,1)
g_e = find_max(g,382,555,1)
g_f = find_max(g,607,780,1)
g_g = find_max(g,846,1019,1)
g_h = find_max(g,1075,1248,1)
g_i = find_max(g,1300,1473,1)
g_j = find_max(g,1539,1712,1)


r_g_c = np.sqrt(g[38:44+1]/(44-38+1))

b12_c = find_max(b12,37,44,1)
b12_d = find_max(b12,174,293,1)
b12_e = find_max(b12,382,555,1)
b12_f = find_max(b12,607,780,1)
b12_g = find_max(b12,846,1019,1)
b12_h = find_max(b12,1075,1248,1)
b12_i = find_max(b12,1300,1473,1)
b12_j = find_max(b12,1539,1712,1)

b60_c = find_max(b60,37,44,1)
b60_d = find_max(b60,174,293,1)
b60_e = find_max(b60,382,555,1)
b60_f = find_max(b60,607,780,1)
b60_g = find_max(b60,846,1019,1)
b60_h = find_max(b60,1075,1248,1)
b60_i = find_max(b60,1300,1473,1)
b60_j = find_max(b60,1539,1712,1)




















