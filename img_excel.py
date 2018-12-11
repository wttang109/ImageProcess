# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:16:53 2018

@author: sunny
"""
import cv2
import numpy as np
#import xlwt
import pandas as pd
import os

fileList = []
folderCount = 0
rootdir = 'D:\\1205_BTF\\BMP\\minus'
col_num = 2
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        f = os.path.join(root,file)
        fileList.append(f)

for i in range(0,len(fileList)):
    print('Loading files [{x}/{y}]'.format(x=i+1,y=len(fileList)))
    test = fileList[i]
    img = cv2.imread(test,0)

    rows,cols = img.shape[:2]
#    rows_hlf = int(rows/2)
    rows_h = 300   # cut the empty of front and end
    rows_l = 6600
    
    sam = rows_l-rows_h
    sam_hlf = int(sam/2)
    step = 0.0423
    sam_rate = 1/step
    dist = sam_rate/sam
    del_f = []
    for i in range(0,sam):
        dist_list = dist*i
        del_f.append(dist_list)
    
    '''
    def col_fft_abs(test_image,x_pixel,cn):
        col_x = np.zeros(shape=(rows,cn))
        f_x = np.zeros(shape=(rows,cn))
        s_x = np.zeros(shape=(rows,cn))
        for i in range(x_pixel-int(cn/2),x_pixel+int(cn/2)):
            col_x[:,i-x_pixel+int(cn/2)] = img[:,i]
            f_x[:,i-x_pixel+int(cn/2)] = np.fft.fft(col_x[:,i-x_pixel+int(cn/2)])
            s_x[:,i-x_pixel+int(cn/2)] = np.abs(f_x[:,i-x_pixel+int(cn/2)])
        return col_x, s_x
    '''
    def col_fft_abs(col_x,x_pixel):
        col_x = img[rows_h:rows_l,x_pixel]
        f_x = np.fft.fft(col_x)
        s_x = np.abs(f_x)
        return col_x, s_x

    col_45k, s_45k  = col_fft_abs(img,500)
    col_15c, s_15c  = col_fft_abs(img,780)
    col_30m, s_30m  = col_fft_abs(img,1040)
    col_90y, s_90y  = col_fft_abs(img,1290)
 
    col_50,  s_50   = col_fft_abs(img,1530)
    col_75,  s_75   = col_fft_abs(img,1930)
    col_100, s_100  = col_fft_abs(img,2330)
    col_50L, s_50L  = col_fft_abs(img,2740)
    col_75L, s_75L  = col_fft_abs(img,3140)
    col_100L,s_100L = col_fft_abs(img,3550)

    col_R,   s_R    = col_fft_abs(img,3940)
    col_G,   s_G    = col_fft_abs(img,4140)
    col_B,   s_B    = col_fft_abs(img,4330)
    col_BL,  s_BL   = col_fft_abs(img,4520)
    
    listk = ['del_f',
             's_45k','s_15c','s_30m','s_90y',
             's_50', 's_75', 's_100',
             's_50L','s_75L','s_100L',
             's_R',  's_G',  's_B',  's_BL']
    datas = {}
    datas['del_f']  = del_f[:3150]
    
    datas['s_45k']  = s_45k[:3150].tolist()
    datas['s_15c']  = s_15c[:3150].tolist()
    datas['s_30m']  = s_30m[:3150].tolist()
    datas['s_90y']  = s_90y[:3150].tolist()
    
    datas['s_50']   = s_50[:3150].tolist()
    datas['s_75']   = s_75[:3150].tolist()
    datas['s_100']  = s_100[:3150].tolist()
    datas['s_50L']  = s_50L[:3150].tolist()
    datas['s_75L']  = s_75L[:3150].tolist()
    datas['s_100L'] = s_100L[:3150].tolist()
    
    datas['s_R']    = s_R[:3150].tolist()
    datas['s_G']    = s_G[:3150].tolist()
    datas['s_B']    = s_B[:3150].tolist()
    datas['s_BL']   = s_BL[:3150].tolist()

    cols = pd.DataFrame(columns = listk)

    for id in listk:
        cols[id] = datas[id]
    test_split = test.split('\\').pop().split('/').pop().replace('MFP45_600C_','').rsplit('.', 1)[0]
    cols.to_csv(rootdir+'\\FFT_{x}.csv'.format(x=test_split))

'''
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
'''


