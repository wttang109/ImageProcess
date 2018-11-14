# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:42:14 2018

@author: User
"""
import numpy as np
import pandas as pd

file_name = 'data.xlsx'
xl_workbook = pd.ExcelFile(file_name)  # Load the excel workbook
df = xl_workbook.parse("Sheet1")  # Parse the sheet into a dataframe
bg_150 = df['bg1_150lppi_L'].tolist()  # Cast the desired column into a python list
bg_100 = df['bg1_100lppi_L'].tolist()
g2_150 = df['g2_150lppi_L'].tolist()
g2_100 = df['g2_100lppi_L'].tolist()

f_g2_150 = np.fft.fft(g2_150[300:6700])
f_g2_100 = np.fft.fft(g2_100[300:6700])
f_bg_150 = np.fft.fft(bg_150[300:6700])
f_bg_100 = np.fft.fft(bg_100[300:6700])

a_g2_150 = np.abs(f_g2_150)
a_g2_100 = np.abs(f_g2_100)
a_bg_150 = np.abs(f_bg_150)
a_bg_100 = np.abs(f_bg_100)

sam = 6400
step = 0.0423
sam_rate = 1/step
dist = sam_rate/sam
del_f = []
for i in range(0,6400):
    dist_list = dist*i
    del_f.append(dist_list)

######### https://blog.csdn.net/u013250416/article/details/53189019
listk = ['del_frequency','g2_150','g2_100','bg_150','bg_100']
datas = {}
datas['del_frequency'] = del_f
datas['g2_150'] = a_g2_150
datas['g2_100'] = a_g2_100.tolist()
datas['bg_150'] = a_bg_150.tolist()
datas['bg_100'] = a_bg_100.tolist()

cols = pd.DataFrame(columns = listk)

for id in listk:
    cols[id] = datas[id]
cols.to_csv('g2_bg_100_150lppi.csv')

