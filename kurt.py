# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:14:09 2018

@author: User
"""

import numpy as np
import pandas as pd

file_name = 'ft_kurt.xlsx'
xl_workbook = pd.ExcelFile(file_name)  # Load the excel workbook
df = xl_workbook.parse("Plot Values_FFT")  # Parse the sheet into a dataframe
delf = np.array(df['Frequency'].tolist())
g    = np.array(df['100lppi_L_good'].tolist())  # Cast the desired column into a python list
b12  = np.array(df['100lppi_L_geardefect_12T'].tolist())
b60  = np.array(df['100lppi_L_brokengear60T'].tolist())

def find_max(col,left,right,maxnum):
#    max_abs = col[left:right][(np.argsort(col[left:right])[-maxnum:])]
#    rms = np.sqrt(sum(np.square(g[left:right+1]))/(right-left+1))
#    cf = max_abs/rms
    ks = pd.Series(col[left:right+1])
    k = ks.kurt()
    return k

k_g_c = find_max(g,38,44,1)
k_g_d = find_max(g,174,293,1)
k_g_e = find_max(g,382,555,1)
k_g_f = find_max(g,607,780,1)
k_g_g = find_max(g,846,1019,1)
k_g_h = find_max(g,1075,1248,1)
k_g_i = find_max(g,1300,1473,1)
k_g_j = find_max(g,1539,1712,1)

k_b12_c = find_max(b12,38,44,1)
k_b12_d = find_max(b12,174,293,1)
k_b12_e = find_max(b12,382,555,1)
k_b12_f = find_max(b12,607,780,1)
k_b12_g = find_max(b12,846,1019,1)
k_b12_h = find_max(b12,1075,1248,1)
k_b12_i = find_max(b12,1300,1473,1)
k_b12_j = find_max(b12,1539,1712,1)

k_b60_c = find_max(b60,38,44,1)
k_b60_d = find_max(b60,174,293,1)
k_b60_e = find_max(b60,382,555,1)
k_b60_f = find_max(b60,607,780,1)
k_b60_g = find_max(b60,846,1019,1)
k_b60_h = find_max(b60,1075,1248,1)
k_b60_i = find_max(b60,1300,1473,1)
k_b60_j = find_max(b60,1539,1712,1)




















