# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:16:53 2018

@author: User
"""
import cv2
import numpy as np
import xlwt
import pandas as pd
test1 = 'MFP_good_1_MFP_good_2'
test2 = 'MFP_good_1_MFP_12T-defect-gear_1'
test3 = 'MFP_good_1_MFP_60T-brokengear_1'

img1 = cv2.imread('{x}.bmp'.format(x=test1),0)
img2 = cv2.imread('{x}.bmp'.format(x=test2),0)
img3 = cv2.imread('{x}.bmp'.format(x=test3),0)
#a = np.array([[1,2,3],[4,5,6]])
#np.savetxt('new.csv', img)
L1_150 = img1[:,1440]
L1_100 = img1[:,2185]
L1_75 = img1[:,2910]
L1_50 = img1[:,3660]

L2_150 = img2[:,1440]
L2_100 = img2[:,2185]
L2_75 = img2[:,2910]
L2_50 = img2[:,3660]

L3_150 = img3[:,1440]
L3_100 = img3[:,2185]
L3_75 = img3[:,2910]
L3_50 = img3[:,3660]

listk = ['g1g2_150','g1g2_100','g1g2_75','g1g2_50',
         'g1_12T_150','g1_12T_100','g1_12T_75','g1_12T_50',
         'g1_60T_150','g1_60T_100','g1_60T_75','g1_60T_50']
datas = {}
datas['g1g2_150'] = L1_150.tolist()
datas['g1g2_100'] = L1_100.tolist()
datas['g1g2_75'] = L1_75.tolist()
datas['g1g2_50'] = L1_50.tolist()

datas['g1_12T_150'] = L2_150.tolist()
datas['g1_12T_100'] = L2_100.tolist()
datas['g1_12T_75'] = L2_75.tolist()
datas['g1_12T_50'] = L2_50.tolist()

datas['g1_60T_150'] = L3_150.tolist()
datas['g1_60T_100'] = L3_100.tolist()
datas['g1_60T_75'] = L3_75.tolist()
datas['g1_60T_50'] = L3_50.tolist()
cols = pd.DataFrame(columns = listk)

for id in listk:
    cols[id] = datas[id]
cols.to_csv('minus1121.csv')



