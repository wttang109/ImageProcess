# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:13:49 2018

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test = 'minus_IT45_defect-gear_total-image_600color.tif'
img = cv2.imread('{x}.tif'.format(x=test),0) 

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

def find_max(col,left,right,maxnum):
    fft = col[left:right][(np.argsort(col[left:right])[-maxnum:])]
    index = np.argsort(col[left:right])[-maxnum:]+left
    plt.scatter(index,fft)
    for a, b in zip(index, fft):
        plt.text(a, b, (a,b), ha='center', va='bottom', fontsize=20)

sam = 6950
step = 0.0423
sam_rate = 1/step
dist = sam_rate/sam
del_f = []
for i in range(0,rows_h):
    dist_list = dist*i
    del_f.append(dist_list)

def plot_f(subnum,y,value,picname,xname):
    plt.subplot(subnum)
    plt.plot(y,value)
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
#    max_indx=np.argmax(ran)#max value index    
#    plt.plot(max_indx,ran[max_indx],'ks')
    
#    show_max='['+str(max_indx最大值位置)+' '+str(ran[max_indx]最大值)+']'
#    plt.annotate(show_max,xytext=(max_indx,ran[max_indx]),xy=(max_indx,ran[max_indx]))
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("FFT_{x}_{y} lppi".format(x=picname,y=xname),fontsize=60)
    plt.ylabel('abs',fontsize=40)
    my_x_ticks = np.arange(0, 12, 0.5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 12000))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_f(311,del_f,s_50[:rows_h],test,'s_50')
plot_f(312,del_f,s_75[:rows_h],test,'s_75')
plot_f(313,del_f,s_100[:rows_h],test,'s_100')
plt.savefig("_test_FFT_50_75_100_150_{x}.png".format(x=test))

#s_50[150:450][sorted(np.argsort(s_50[150:450])[-3:])]  #原來順序np指定區間找前三大值
#np.argsort(x) #由小到大的索引

plt.figure(figsize=(25,15))
plt.style.use('ggplot')
find_max(s_50,150,450,3)
find_max(s_50,700,1100,3)
find_max(s_50,1300,1700,3)
find_max(s_50,1900,2300,3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("_test_max_{x}.png".format(x=test))
plt.show()




















