# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:13:49 2018

@author: sunny
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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
kt_c_150L=[]
kt_d_150L=[]
kt_e_150L=[]
kt_f_150L=[]
kt_g_150L=[]
kt_h_150L=[]
kt_i_150L=[]
kt_j_150L=[]
sk_c_150L=[]
sk_d_150L=[]
sk_e_150L=[]
sk_f_150L=[]
sk_g_150L=[]
sk_h_150L=[]
sk_i_150L=[]
sk_j_150L=[]

kt_c_100L=[]
kt_d_100L=[]
kt_e_100L=[]
kt_f_100L=[]
kt_g_100L=[]
kt_h_100L=[]
kt_i_100L=[]
kt_j_100L=[]
sk_c_100L=[]
sk_d_100L=[]
sk_e_100L=[]
sk_f_100L=[]
sk_g_100L=[]
sk_h_100L=[]
sk_i_100L=[]
sk_j_100L=[]

kt_c_75L=[]
kt_d_75L=[]
kt_e_75L=[]
kt_f_75L=[]
kt_g_75L=[]
kt_h_75L=[]
kt_i_75L=[]
kt_j_75L=[]
sk_c_75L=[]
sk_d_75L=[]
sk_e_75L=[]
sk_f_75L=[]
sk_g_75L=[]
sk_h_75L=[]
sk_i_75L=[]
sk_j_75L=[]

label_list = []
################ take N colums for fft sample ##################################################

fileList = []
folderCount = 0
rootdir = 'eut'
col_num = 100
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        f = os.path.join(root,file)
        #print(f)
        fileList.append(f)
        
for i in range(1,len(fileList)):
    print('Loading files [{x}/{y}]'.format(x=i,y=len(fileList)-1))
    test = fileList[i]
    img = cv2.imread(test,0)
    if '12T' in test:
        label = '12T'
    elif '60T' in test:
        label = '60T'
    elif test.count('good')==2:
        label = 'good'

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

    def kurt(cn,col,left,right,kt,sk):
        for i in range(0,cn):
            ks = pd.Series(col[:,i][left:right+1])
            k = ks.kurt()
            s = ks.skew()
            kt.append(k)
            sk.append(s)
            
    for x in range(0,col_num):
        label_list.append(label)
    
    print('Find kurtosis and skewness')
    left  = [150,740, 1185,1631,1998,2518,2887,3332]
    right = [600,1037,1630,1925,2370,2815,3331,3480]

    kurt(col_num,s_150L,left[0],right[0],kt_c_150L,sk_c_150L)
    kurt(col_num,s_150L,left[1],right[1],kt_d_150L,sk_d_150L)
    kurt(col_num,s_150L,left[2],right[2],kt_e_150L,sk_e_150L)
    kurt(col_num,s_150L,left[3],right[3],kt_f_150L,sk_f_150L)
    kurt(col_num,s_150L,left[4],right[4],kt_g_150L,sk_g_150L)
    kurt(col_num,s_150L,left[5],right[5],kt_h_150L,sk_h_150L)
    kurt(col_num,s_150L,left[6],right[6],kt_i_150L,sk_i_150L)
    kurt(col_num,s_150L,left[7],right[7],kt_j_150L,sk_j_150L)

    kurt(col_num,s_100L,left[0],right[0],kt_c_100L,sk_c_100L)
    kurt(col_num,s_100L,left[1],right[1],kt_d_100L,sk_d_100L)
    kurt(col_num,s_100L,left[2],right[2],kt_e_100L,sk_e_100L)
    kurt(col_num,s_100L,left[3],right[3],kt_f_100L,sk_f_100L)
    kurt(col_num,s_100L,left[4],right[4],kt_g_100L,sk_g_100L)
    kurt(col_num,s_100L,left[5],right[5],kt_h_100L,sk_h_100L)
    kurt(col_num,s_100L,left[6],right[6],kt_i_100L,sk_i_100L)
    kurt(col_num,s_100L,left[7],right[7],kt_j_100L,sk_j_100L)

    kurt(col_num,s_75L,left[0],right[0],kt_c_75L,sk_c_75L)
    kurt(col_num,s_75L,left[1],right[1],kt_d_75L,sk_d_75L)
    kurt(col_num,s_75L,left[2],right[2],kt_e_75L,sk_e_75L)
    kurt(col_num,s_75L,left[3],right[3],kt_f_75L,sk_f_75L)
    kurt(col_num,s_75L,left[4],right[4],kt_g_75L,sk_g_75L)
    kurt(col_num,s_75L,left[5],right[5],kt_h_75L,sk_h_75L)
    kurt(col_num,s_75L,left[6],right[6],kt_i_75L,sk_i_75L)
    kurt(col_num,s_75L,left[7],right[7],kt_j_75L,sk_j_75L)

print('Write data to csv')
listks = ['kt_c_150L','kt_d_150L','kt_e_150L','kt_f_150L','kt_g_150L','kt_h_150L','kt_i_150L','kt_j_150L',
          'kt_c_100L','kt_d_100L','kt_e_100L','kt_f_100L','kt_g_100L','kt_h_100L','kt_i_100L','kt_j_100L',
          'kt_c_75L', 'kt_d_75L', 'kt_e_75L', 'kt_f_75L', 'kt_g_75L', 'kt_h_75L', 'kt_i_75L', 'kt_j_75L',
          'sk_c_150L','sk_d_150L','sk_e_150L','sk_f_150L','sk_g_150L','sk_h_150L','sk_i_150L','sk_j_150L',
          'sk_c_100L','sk_d_100L','sk_e_100L','sk_f_100L','sk_g_100L','sk_h_100L','sk_i_100L','sk_j_100L',
          'sk_c_75L', 'sk_d_75L', 'sk_e_75L', 'sk_f_75L', 'sk_g_75L', 'sk_h_75L', 'sk_i_75L', 'sk_j_75L','label_list']
datas = {}
datas['kt_c_150L'] = kt_c_150L
datas['kt_d_150L'] = kt_d_150L
datas['kt_e_150L'] = kt_e_150L
datas['kt_f_150L'] = kt_f_150L
datas['kt_g_150L'] = kt_g_150L
datas['kt_h_150L'] = kt_h_150L
datas['kt_i_150L'] = kt_i_150L
datas['kt_j_150L'] = kt_j_150L

datas['kt_c_100L'] = kt_c_100L
datas['kt_d_100L'] = kt_d_100L
datas['kt_e_100L'] = kt_e_100L
datas['kt_f_100L'] = kt_f_100L
datas['kt_g_100L'] = kt_g_100L
datas['kt_h_100L'] = kt_h_100L
datas['kt_i_100L'] = kt_i_100L
datas['kt_j_100L'] = kt_j_100L

datas['kt_c_75L'] = kt_c_75L
datas['kt_d_75L'] = kt_d_75L
datas['kt_e_75L'] = kt_e_75L
datas['kt_f_75L'] = kt_f_75L
datas['kt_g_75L'] = kt_g_75L
datas['kt_h_75L'] = kt_h_75L
datas['kt_i_75L'] = kt_i_75L
datas['kt_j_75L'] = kt_j_75L

datas['sk_c_150L'] = sk_c_150L
datas['sk_d_150L'] = sk_d_150L
datas['sk_e_150L'] = sk_e_150L
datas['sk_f_150L'] = sk_f_150L
datas['sk_g_150L'] = sk_g_150L
datas['sk_h_150L'] = sk_h_150L
datas['sk_i_150L'] = sk_i_150L
datas['sk_j_150L'] = sk_j_150L

datas['sk_c_100L'] = sk_c_100L
datas['sk_d_100L'] = sk_d_100L
datas['sk_e_100L'] = sk_e_100L
datas['sk_f_100L'] = sk_f_100L
datas['sk_g_100L'] = sk_g_100L
datas['sk_h_100L'] = sk_h_100L
datas['sk_i_100L'] = sk_i_100L
datas['sk_j_100L'] = sk_j_100L

datas['sk_c_75L'] = sk_c_75L
datas['sk_d_75L'] = sk_d_75L
datas['sk_e_75L'] = sk_e_75L
datas['sk_f_75L'] = sk_f_75L
datas['sk_g_75L'] = sk_g_75L
datas['sk_h_75L'] = sk_h_75L
datas['sk_i_75L'] = sk_i_75L
datas['sk_j_75L'] = sk_j_75L
datas['label_list'] = label_list # label of good or defect

ex_cols = pd.DataFrame(columns = listks)
for id in listks:
    ex_cols[id] = datas[id]
ex_cols.to_csv('kurt_skew.csv')

'''
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
    plt.ylim((0, 8000))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plt.style.use('ggplot')
plot_f(311,del_f,s_150L[:rows_h,:],test,'s_150L')
plot_f(312,del_f,s_100L[:rows_h,:],test,'s_100L')
plot_f(313,del_f,s_75L[:rows_h,:],test,'s_75L')
plt.savefig("FFT_{x}.png".format(x=test))

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
'''





