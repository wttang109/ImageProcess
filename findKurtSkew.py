# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:32:33 2018

@author: Sunny
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

fileList = []
folderCount = 0
rootdir = 'D:\\toexcel'
col_num = 100
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        f = os.path.join(root,file)
        #print(f)
        fileList.append(f)

kt_0_50=[]        
kt_1_50=[]
kt_2_50=[]
kt_3_50=[]
kt_4_50=[]
kt_5_50=[]
kt_6_50=[]
kt_7_50=[]
kt_8_50=[]
kt_9_50=[]
sk_0_50=[]
sk_1_50=[]
sk_2_50=[]
sk_3_50=[]
sk_4_50=[]
sk_5_50=[]
sk_6_50=[]
sk_7_50=[]
sk_8_50=[]
sk_9_50=[]

kt_0_75=[]
kt_1_75=[]
kt_2_75=[]
kt_3_75=[]
kt_4_75=[]
kt_5_75=[]
kt_6_75=[]
kt_7_75=[]
kt_8_75=[]
kt_9_75=[]
sk_0_75=[]
sk_1_75=[]
sk_2_75=[]
sk_3_75=[]
sk_4_75=[]
sk_5_75=[]
sk_6_75=[]
sk_7_75=[]
sk_8_75=[]
sk_9_75=[]

kt_0_100=[]
kt_1_100=[]
kt_2_100=[]
kt_3_100=[]
kt_4_100=[]
kt_5_100=[]
kt_6_100=[]
kt_7_100=[]
kt_8_100=[]
kt_9_100=[]
sk_0_100=[]
sk_1_100=[]
sk_2_100=[]
sk_3_100=[]
sk_4_100=[]
sk_5_100=[]
sk_6_100=[]
sk_7_100=[]
sk_8_100=[]
sk_9_100=[]

kt_0_50L=[]
kt_1_50L=[]
kt_2_50L=[]
kt_3_50L=[]
kt_4_50L=[]
kt_5_50L=[]
kt_6_50L=[]
kt_7_50L=[]
kt_8_50L=[]
kt_9_50L=[]
sk_0_50L=[]
sk_1_50L=[]
sk_2_50L=[]
sk_3_50L=[]
sk_4_50L=[]
sk_5_50L=[]
sk_6_50L=[]
sk_7_50L=[]
sk_8_50L=[]
sk_9_50L=[]

kt_0_75L=[]
kt_1_75L=[]
kt_2_75L=[]
kt_3_75L=[]
kt_4_75L=[]
kt_5_75L=[]
kt_6_75L=[]
kt_7_75L=[]
kt_8_75L=[]
kt_9_75L=[]
sk_0_75L=[]
sk_1_75L=[]
sk_2_75L=[]
sk_3_75L=[]
sk_4_75L=[]
sk_5_75L=[]
sk_6_75L=[]
sk_7_75L=[]
sk_8_75L=[]
sk_9_75L=[]

kt_0_100L=[]
kt_1_100L=[]
kt_2_100L=[]
kt_3_100L=[]
kt_4_100L=[]
kt_5_100L=[]
kt_6_100L=[]
kt_7_100L=[]
kt_8_100L=[]
kt_9_100L=[]
sk_0_100L=[]
sk_1_100L=[]
sk_2_100L=[]
sk_3_100L=[]
sk_4_100L=[]
sk_5_100L=[]
sk_6_100L=[]
sk_7_100L=[]
sk_8_100L=[]
sk_9_100L=[]

label_list = []

for i in range(0,len(fileList)):
    print('Loading files [{x}/{y}]'.format(x=i+1,y=len(fileList)))
    test = fileList[i]
    img = cv2.imread(test,0)
    
    if 'no-grease' in test:
        label = 'no-grease'
    elif 'grease-1' in test:
        label = 'grease-1'
    elif 'grease-2' in test:
        label = 'grease-2'
    elif test.count('grease-gear')==2:
        label = 'good'
    
    rows,cols = img.shape[:2]
    rows_hlf = int(rows/2)
    rows_h = 300   # cut the empty of front and end
    rows_l = 6600
    def col_fft_abs(test_image,x_pixel,cn):
        col_x = np.zeros(shape=(rows_l-rows_h,cn))
        f_x = np.zeros(shape=(rows_l-rows_h,cn))
        s_x = np.zeros(shape=(rows_l-rows_h,cn))
        for i in range(x_pixel-int(cn/2),x_pixel+int(cn/2)):
            col_x[:,i-x_pixel+int(cn/2)] = img[rows_h:rows_l,i]
            f_x[:,i-x_pixel+int(cn/2)] = np.fft.fft(col_x[:,i-x_pixel+int(cn/2)])
            s_x[:,i-x_pixel+int(cn/2)] = np.abs(f_x[:,i-x_pixel+int(cn/2)])
        return col_x, s_x
    
    col_50,  s_50   = col_fft_abs(img,1530,col_num)
    col_75,  s_75   = col_fft_abs(img,1930,col_num)
    col_100, s_100  = col_fft_abs(img,2330,col_num)
    col_50L, s_50L  = col_fft_abs(img,2740,col_num)
    col_75L, s_75L  = col_fft_abs(img,3140,col_num)
    col_100L,s_100L = col_fft_abs(img,3550,col_num)

    def kurtSkew(cn,col,left,right,kt,sk):
        for i in range(0,cn):
            ks = pd.Series(col[rows_h:rows_l,i][left:right+1])
            k = ks.kurt()
            s = ks.skew()
            kt.append(k)
            sk.append(s)
            
    for x in range(0,col_num):
        label_list.append(label)
    
    print('Find kurtosis and skewness')
    left  = [446,743, 1041,1337,1634,2004,2301,2598,2895,3191]
    right = [742,1040,1336,1633,2003,2300,2597,2894,3190,3488]

    kurtSkew(col_num,s_50,left[0],right[0],kt_0_50,sk_0_50)
    kurtSkew(col_num,s_50,left[1],right[1],kt_1_50,sk_1_50)
    kurtSkew(col_num,s_50,left[2],right[2],kt_2_50,sk_2_50)
    kurtSkew(col_num,s_50,left[3],right[3],kt_3_50,sk_3_50)
    kurtSkew(col_num,s_50,left[4],right[4],kt_4_50,sk_4_50)
    kurtSkew(col_num,s_50,left[5],right[5],kt_5_50,sk_5_50)
    kurtSkew(col_num,s_50,left[6],right[6],kt_6_50,sk_6_50)
    kurtSkew(col_num,s_50,left[7],right[7],kt_7_50,sk_7_50)
    kurtSkew(col_num,s_50,left[8],right[8],kt_8_50,sk_8_50)
    kurtSkew(col_num,s_50,left[9],right[9],kt_9_50,sk_9_50)
        
    kurtSkew(col_num,s_75,left[0],right[0],kt_0_75,sk_0_75)
    kurtSkew(col_num,s_75,left[1],right[1],kt_1_75,sk_1_75)
    kurtSkew(col_num,s_75,left[2],right[2],kt_2_75,sk_2_75)
    kurtSkew(col_num,s_75,left[3],right[3],kt_3_75,sk_3_75)
    kurtSkew(col_num,s_75,left[4],right[4],kt_4_75,sk_4_75)
    kurtSkew(col_num,s_75,left[5],right[5],kt_5_75,sk_5_75)
    kurtSkew(col_num,s_75,left[6],right[6],kt_6_75,sk_6_75)
    kurtSkew(col_num,s_75,left[7],right[7],kt_7_75,sk_7_75)
    kurtSkew(col_num,s_75,left[8],right[8],kt_8_75,sk_8_75)
    kurtSkew(col_num,s_75,left[9],right[9],kt_9_75,sk_9_75)
        
    kurtSkew(col_num,s_100,left[0],right[0],kt_0_100,sk_0_100)
    kurtSkew(col_num,s_100,left[1],right[1],kt_1_100,sk_1_100)
    kurtSkew(col_num,s_100,left[2],right[2],kt_2_100,sk_2_100)
    kurtSkew(col_num,s_100,left[3],right[3],kt_3_100,sk_3_100)
    kurtSkew(col_num,s_100,left[4],right[4],kt_4_100,sk_4_100)
    kurtSkew(col_num,s_100,left[5],right[5],kt_5_100,sk_5_100)
    kurtSkew(col_num,s_100,left[6],right[6],kt_6_100,sk_6_100)
    kurtSkew(col_num,s_100,left[7],right[7],kt_7_100,sk_7_100)
    kurtSkew(col_num,s_100,left[8],right[8],kt_8_100,sk_8_100)
    kurtSkew(col_num,s_100,left[9],right[9],kt_9_100,sk_9_100)

    kurtSkew(col_num,s_50L,left[0],right[0],kt_0_50L,sk_0_50L)
    kurtSkew(col_num,s_50L,left[1],right[1],kt_1_50L,sk_1_50L)
    kurtSkew(col_num,s_50L,left[2],right[2],kt_2_50L,sk_2_50L)
    kurtSkew(col_num,s_50L,left[3],right[3],kt_3_50L,sk_3_50L)
    kurtSkew(col_num,s_50L,left[4],right[4],kt_4_50L,sk_4_50L)
    kurtSkew(col_num,s_50L,left[5],right[5],kt_5_50L,sk_5_50L)
    kurtSkew(col_num,s_50L,left[6],right[6],kt_6_50L,sk_6_50L)
    kurtSkew(col_num,s_50L,left[7],right[7],kt_7_50L,sk_7_50L)
    kurtSkew(col_num,s_50L,left[8],right[8],kt_8_50L,sk_8_50L)
    kurtSkew(col_num,s_50L,left[9],right[9],kt_9_50L,sk_9_50L)
        
    kurtSkew(col_num,s_75L,left[0],right[0],kt_0_75L,sk_0_75L)
    kurtSkew(col_num,s_75L,left[1],right[1],kt_1_75L,sk_1_75L)
    kurtSkew(col_num,s_75L,left[2],right[2],kt_2_75L,sk_2_75L)
    kurtSkew(col_num,s_75L,left[3],right[3],kt_3_75L,sk_3_75L)
    kurtSkew(col_num,s_75L,left[4],right[4],kt_4_75L,sk_4_75L)
    kurtSkew(col_num,s_75L,left[5],right[5],kt_5_75L,sk_5_75L)
    kurtSkew(col_num,s_75L,left[6],right[6],kt_6_75L,sk_6_75L)
    kurtSkew(col_num,s_75L,left[7],right[7],kt_7_75L,sk_7_75L)
    kurtSkew(col_num,s_75L,left[8],right[8],kt_8_75L,sk_8_75L)
    kurtSkew(col_num,s_75L,left[9],right[9],kt_9_75L,sk_9_75L)
        
    kurtSkew(col_num,s_100L,left[0],right[0],kt_0_100L,sk_0_100L)
    kurtSkew(col_num,s_100L,left[1],right[1],kt_1_100L,sk_1_100L)
    kurtSkew(col_num,s_100L,left[2],right[2],kt_2_100L,sk_2_100L)
    kurtSkew(col_num,s_100L,left[3],right[3],kt_3_100L,sk_3_100L)
    kurtSkew(col_num,s_100L,left[4],right[4],kt_4_100L,sk_4_100L)
    kurtSkew(col_num,s_100L,left[5],right[5],kt_5_100L,sk_5_100L)
    kurtSkew(col_num,s_100L,left[6],right[6],kt_6_100L,sk_6_100L)
    kurtSkew(col_num,s_100L,left[7],right[7],kt_7_100L,sk_7_100L)
    kurtSkew(col_num,s_100L,left[8],right[8],kt_8_100L,sk_8_100L)
    kurtSkew(col_num,s_100L,left[9],right[9],kt_9_100L,sk_9_100L)

print('Write data to csv')
listks = ['kt_0_50', 'kt_1_50', 'kt_2_50', 'kt_3_50', 'kt_4_50', 'kt_5_50', 'kt_6_50', 'kt_7_50', 'kt_8_50', 'kt_9_50',
          'kt_0_75', 'kt_1_75', 'kt_2_75', 'kt_3_75', 'kt_4_75', 'kt_5_75', 'kt_6_75', 'kt_7_75', 'kt_8_75', 'kt_9_75',
          'kt_0_100','kt_1_100','kt_2_100','kt_3_100','kt_4_100','kt_5_100','kt_6_100','kt_7_100','kt_8_100','kt_9_100',
          'kt_0_50L', 'kt_1_50L', 'kt_2_50L', 'kt_3_50L', 'kt_4_50L', 'kt_5_50L', 'kt_6_50L', 'kt_7_50L', 'kt_8_50L', 'kt_9_50L',
          'kt_0_75L', 'kt_1_75L', 'kt_2_75L', 'kt_3_75L', 'kt_4_75L', 'kt_5_75L', 'kt_6_75L', 'kt_7_75L', 'kt_8_75L', 'kt_9_75L',
          'kt_0_100L','kt_1_100L','kt_2_100L','kt_3_100L','kt_4_100L','kt_5_100L','kt_6_100L','kt_7_100L','kt_8_100L','kt_9_100L',
          'sk_0_50', 'sk_1_50', 'sk_2_50', 'sk_3_50', 'sk_4_50', 'sk_5_50', 'sk_6_50', 'sk_7_50', 'sk_8_50', 'sk_9_50',
          'sk_0_75', 'sk_1_75', 'sk_2_75', 'sk_3_75', 'sk_4_75', 'sk_5_75', 'sk_6_75', 'sk_7_75', 'sk_8_75', 'sk_9_75',
          'sk_0_100','sk_1_100','sk_2_100','sk_3_100','sk_4_100','sk_5_100','sk_6_100','sk_7_100','sk_8_100','sk_9_100',
          'sk_0_50L', 'sk_1_50L', 'sk_2_50L', 'sk_3_50L', 'sk_4_50L', 'sk_5_50L', 'sk_6_50L', 'sk_7_50L', 'sk_8_50L', 'sk_9_50L',
          'sk_0_75L', 'sk_1_75L', 'sk_2_75L', 'sk_3_75L', 'sk_4_75L', 'sk_5_75L', 'sk_6_75L', 'sk_7_75L', 'sk_8_75L', 'sk_9_75L',
          'sk_0_100L','sk_1_100L','sk_2_100L','sk_3_100L','sk_4_100L','sk_5_100L','sk_6_100L','sk_7_100L','sk_8_100L','sk_9_100L','label_list']
datas = {}
datas['kt_0_50'] = kt_0_50
datas['kt_1_50'] = kt_1_50
datas['kt_2_50'] = kt_2_50
datas['kt_3_50'] = kt_3_50
datas['kt_4_50'] = kt_4_50
datas['kt_5_50'] = kt_5_50
datas['kt_6_50'] = kt_6_50
datas['kt_7_50'] = kt_7_50
datas['kt_8_50'] = kt_8_50
datas['kt_9_50'] = kt_9_50
datas['kt_0_75'] = kt_0_75
datas['kt_1_75'] = kt_1_75
datas['kt_2_75'] = kt_2_75
datas['kt_3_75'] = kt_3_75
datas['kt_4_75'] = kt_4_75
datas['kt_5_75'] = kt_5_75
datas['kt_6_75'] = kt_6_75
datas['kt_7_75'] = kt_7_75
datas['kt_8_75'] = kt_8_75
datas['kt_9_75'] = kt_9_75
datas['kt_0_100'] = kt_0_100
datas['kt_1_100'] = kt_1_100
datas['kt_2_100'] = kt_2_100
datas['kt_3_100'] = kt_3_100
datas['kt_4_100'] = kt_4_100
datas['kt_5_100'] = kt_5_100
datas['kt_6_100'] = kt_6_100
datas['kt_7_100'] = kt_7_100
datas['kt_8_100'] = kt_8_100
datas['kt_9_100'] = kt_9_100

datas['kt_0_50L'] = kt_0_50L
datas['kt_1_50L'] = kt_1_50L
datas['kt_2_50L'] = kt_2_50L
datas['kt_3_50L'] = kt_3_50L
datas['kt_4_50L'] = kt_4_50L
datas['kt_5_50L'] = kt_5_50L
datas['kt_6_50L'] = kt_6_50L
datas['kt_7_50L'] = kt_7_50L
datas['kt_8_50L'] = kt_8_50L
datas['kt_9_50L'] = kt_9_50L
datas['kt_0_75L'] = kt_0_75L
datas['kt_1_75L'] = kt_1_75L
datas['kt_2_75L'] = kt_2_75L
datas['kt_3_75L'] = kt_3_75L
datas['kt_4_75L'] = kt_4_75L
datas['kt_5_75L'] = kt_5_75L
datas['kt_6_75L'] = kt_6_75L
datas['kt_7_75L'] = kt_7_75L
datas['kt_8_75L'] = kt_8_75L
datas['kt_9_75L'] = kt_9_75L
datas['kt_0_100L'] = kt_0_100L
datas['kt_1_100L'] = kt_1_100L
datas['kt_2_100L'] = kt_2_100L
datas['kt_3_100L'] = kt_3_100L
datas['kt_4_100L'] = kt_4_100L
datas['kt_5_100L'] = kt_5_100L
datas['kt_6_100L'] = kt_6_100L
datas['kt_7_100L'] = kt_7_100L
datas['kt_8_100L'] = kt_8_100L
datas['kt_9_100L'] = kt_9_100L

datas['sk_0_50'] = sk_0_50
datas['sk_1_50'] = sk_1_50
datas['sk_2_50'] = sk_2_50
datas['sk_3_50'] = sk_3_50
datas['sk_4_50'] = sk_4_50
datas['sk_5_50'] = sk_5_50
datas['sk_6_50'] = sk_6_50
datas['sk_7_50'] = sk_7_50
datas['sk_8_50'] = sk_8_50
datas['sk_9_50'] = sk_9_50
datas['sk_0_75'] = sk_0_75
datas['sk_1_75'] = sk_1_75
datas['sk_2_75'] = sk_2_75
datas['sk_3_75'] = sk_3_75
datas['sk_4_75'] = sk_4_75
datas['sk_5_75'] = sk_5_75
datas['sk_6_75'] = sk_6_75
datas['sk_7_75'] = sk_7_75
datas['sk_8_75'] = sk_8_75
datas['sk_9_75'] = sk_9_75
datas['sk_0_100'] = sk_0_100
datas['sk_1_100'] = sk_1_100
datas['sk_2_100'] = sk_2_100
datas['sk_3_100'] = sk_3_100
datas['sk_4_100'] = sk_4_100
datas['sk_5_100'] = sk_5_100
datas['sk_6_100'] = sk_6_100
datas['sk_7_100'] = sk_7_100
datas['sk_8_100'] = sk_8_100
datas['sk_9_100'] = sk_9_100

datas['sk_0_50L'] = sk_0_50L
datas['sk_1_50L'] = sk_1_50L
datas['sk_2_50L'] = sk_2_50L
datas['sk_3_50L'] = sk_3_50L
datas['sk_4_50L'] = sk_4_50L
datas['sk_5_50L'] = sk_5_50L
datas['sk_6_50L'] = sk_6_50L
datas['sk_7_50L'] = sk_7_50L
datas['sk_8_50L'] = sk_8_50L
datas['sk_9_50L'] = sk_9_50L
datas['sk_0_75L'] = sk_0_75L
datas['sk_1_75L'] = sk_1_75L
datas['sk_2_75L'] = sk_2_75L
datas['sk_3_75L'] = sk_3_75L
datas['sk_4_75L'] = sk_4_75L
datas['sk_5_75L'] = sk_5_75L
datas['sk_6_75L'] = sk_6_75L
datas['sk_7_75L'] = sk_7_75L
datas['sk_8_75L'] = sk_8_75L
datas['sk_9_75L'] = sk_9_75L
datas['sk_0_100L'] = sk_0_100L
datas['sk_1_100L'] = sk_1_100L
datas['sk_2_100L'] = sk_2_100L
datas['sk_3_100L'] = sk_3_100L
datas['sk_4_100L'] = sk_4_100L
datas['sk_5_100L'] = sk_5_100L
datas['sk_6_100L'] = sk_6_100L
datas['sk_7_100L'] = sk_7_100L
datas['sk_8_100L'] = sk_8_100L
datas['sk_9_100L'] = sk_9_100L
datas['label_list'] = label_list # label of good or defect

ex_cols = pd.DataFrame(columns = listks)
for id in listks:
    ex_cols[id] = datas[id]
ex_cols.to_csv('D:\\kurt_skew.csv')







