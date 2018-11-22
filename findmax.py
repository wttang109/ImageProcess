# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:13:49 2018

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
test = 'MFP_g1_MFP_b1'
img = cv2.imread('{x}.bmp'.format(x=test),0) 

rows,cols = img.shape[:2]
rows_h = int(rows/2)
def col_fft_abs(test_image,x_pixel):
    col_x = test_image[:,x_pixel]
    f_x = np.fft.fft(col_x[:rows])
    s_x = np.abs(f_x)
    return col_x, s_x

col_50,  s_50   = col_fft_abs(img,1530)
col_75,  s_75   = col_fft_abs(img,1940)
col_100, s_100  = col_fft_abs(img,2330)
'''
################ take N colums for fft sample ##################################################
test = 'MFP_good_1_MFP_60T-brokengear_2'
img = cv2.imread('{x}.bmp'.format(x=test),0) 

rows,cols = img.shape[:2]
rows_h = int(rows/2)
def col_fft_abs(test_image,x_pixel,cn):
    col_x = np.zeros(shape=(rows,cn))
    f_x = np.zeros(shape=(rows,cn))
    s_x = np.zeros(shape=(rows,cn))
    for i in range(x_pixel-int(cn/2),x_pixel+int(cn/2)):
        col_x[:,i-x_pixel+int(cn/2)] = img[:,i]
        f_x[:,i-x_pixel+int(cn/2)] = np.fft.fft(col_x[:,i-x_pixel+int(cn/2)])
        s_x[:,i-x_pixel+int(cn/2)] = np.abs(f_x[:,i-x_pixel+int(cn/2)])
    return col_x, s_x
col_num = 100
'''
col_50,  s_50   = col_fft_abs(img,1530,col_num)
col_75,  s_75   = col_fft_abs(img,1940,col_num)
col_100, s_100  = col_fft_abs(img,2330,col_num)
col_50i, s_50i  = col_fft_abs(img,2760,col_num)
col_75i, s_75i  = col_fft_abs(img,3160,col_num)
col_100i,s_100i = col_fft_abs(img,3550,col_num)
'''
col_150L, s_150L = col_fft_abs(img,1430,col_num)
col_100L, s_100L = col_fft_abs(img,2180,col_num)
col_75L,  s_75L  = col_fft_abs(img,2920,col_num)

################ take N colums for fft sample ##################################################

def find_max(cn,col,left,right,maxnum,f_L,d_L):
    for i in range(0,cn):
        fft = col[:,i][left:right][(np.argsort(col[:,i][left:right])[-maxnum:])] # find max value
        index = np.argsort(col[:,i][left:right])[-maxnum:]+left
        df = index * dist
        for j in range(0,maxnum):
            f_L.append(fft[j])
            d_L.append(df[j])
        plt.scatter(df,fft)
        my_x_ticks = np.arange(0, 12, 0.25)
        plt.xticks(my_x_ticks)
#        for a, b in zip(index, fft):
#            plt.text(a, b, (a,b), ha='center', va='bottom', fontsize=20)

sam = 7000
step = 0.0423
sam_rate = 1/step
dist = sam_rate/sam
del_f = []
for i in range(0,rows_h):
    dist_list = dist*i
    del_f.append(dist_list)
'''
#### plot result ####
def plot_f(subnum,y,value,picname,xname):
    plt.subplot(subnum)
    plt.plot(y,value)
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
#    max_indx=np.argmax(ran)#max value index    
#    plt.plot(max_indx,ran[max_indx],'ks')
    
#    show_max='['+str(max_indx)+' '+str(ran[max_indx])+']'
#    plt.annotate(show_max,xytext=(max_indx,ran[max_indx]),xy=(max_indx,ran[max_indx]))
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.title("FFT_{x}_{y}".format(x=picname,y=xname),fontsize=50)
    plt.ylabel('abs',fontsize=50)
    my_x_ticks = np.arange(0, 12, 0.5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 12000))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plt.style.use('ggplot')
plot_f(311,del_f,s_150L[:rows_h,49],test,'s_150L')
plot_f(312,del_f,s_100L[:rows_h,49],test,'s_100L')
plot_f(313,del_f,s_75L[:rows_h,49],test,'s_75L')
plt.savefig("FFT_{x}.png".format(x=test))
'''
# s_50[150:450][sorted(np.argsort(s_50[150:450])[-3:])]
# np.argsort(x) # sort index
f_150L = []
d_150L = []
f_100L = []
d_100L = []
f_75L = []
d_75L = []

plt.figure(figsize=(40,10))
plt.style.use('ggplot')
find_max(col_num,s_150L,100,1700,3,f_150L,d_150L)
find_max(col_num,s_150L,1900,3350,3,f_150L,d_150L)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("{x}_{y}_{cn}pt".format(x=test,y='s_150L',cn=col_num),fontsize=20)
plt.xlabel('del_f',fontsize=20)
plt.ylabel('abs',fontsize=20)
plt.savefig("MAX2_{x}_{y}_{cn}.png".format(x=test,cn=col_num,y='s_150L'))

plt.figure(figsize=(40,10))
plt.style.use('ggplot')
find_max(col_num,s_100L,100,1100,3,f_100L,d_100L)
find_max(col_num,s_100L,1350,2350,3,f_100L,d_100L)
#find_max(col_num,s_100L,2530,3450,3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("{x}_{y}_{cn}pt".format(x=test,y='s_100L',cn=col_num),fontsize=20)
plt.xlabel('del_f',fontsize=20)
plt.ylabel('abs',fontsize=20)
plt.savefig("MAX2_{x}_{y}_{cn}.png".format(x=test,cn=col_num,y='s_100L'))

plt.figure(figsize=(40,10))
plt.style.use('ggplot')
find_max(col_num,s_100L,100,800,3,f_75L,d_75L)
find_max(col_num,s_100L,1150,1700,3,f_75L,d_75L)
#find_max(col_num,s_100L,1900,2650,3)
#find_max(col_num,s_100L,2750,3330,3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("{x}_{y}_{cn}pt".format(x=test,y='s_75L',cn=col_num),fontsize=20)
plt.xlabel('del_f',fontsize=20)
plt.ylabel('abs',fontsize=20)
plt.savefig("MAX2_{x}_{y}_{cn}.png".format(x=test,cn=col_num,y='s_57L'))
plt.show()

listk = ['f_150L','d_150L','f_100L','d_100L','f_75L','d_75L','type']
datas = {}
datas['f_150L'] = f_150L
datas['d_150L'] = d_150L
datas['f_100L'] = f_100L
datas['d_100L'] = d_100L
datas['f_75L'] = f_75L
datas['d_75L'] = d_75L
datas['type'] = 1 # label of good or defect

cols = pd.DataFrame(columns = listk)
for id in listk:
    cols[id] = datas[id]
cols.to_csv('max_{x}.csv'.format(x=test))






