# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:56:02 2018

@author: sue
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
file_name = 'data.xlsx'
xl_workbook = pd.ExcelFile(file_name)  # Load the excel workbook
df = xl_workbook.parse("Sheet1")  # Parse the sheet into a dataframe
aList = df['50lppi_H'].tolist()  # Cast the desired column into a python list
'''
################# FFT ########################https://blog.csdn.net/on2way/article/details/46981825
#alist = exceltolist.aList #讀取excel資料
#img = cv2.imread('sub_defect_good.bmp') #除3使用
test = 'minus_IT45_defect-gear_total-image_600color.tif'
img = cv2.imread('{x}.tif'.format(x=test),0) 
#cv2.imwrite('img.bmp', img)
'''
col_50 = img[:,1190]  #1190為50l
col_75 = img[:,1925]  #1925為75l
col_100 = img[:,2693] #2693為100l
col_150 = img[:,3437] #3437為150l
'''
rows,cols = img.shape[:2]
rows_h = int(rows/2)
def col_fft_abs(test_image,x_pixel):
#    col_x = test_image[:,x_pixel]
    col_x = np.zeros(shape=(rows,40))
    f_x = np.zeros(shape=(rows,40))
    s_x = np.zeros(shape=(rows,40))
    for i in range(x_pixel-20,x_pixel+20):
        col_x[:,i-x_pixel+20] = img[:,i]
        f_x[:,i-x_pixel+20] = np.fft.fft(col_x[:,i-x_pixel+20])
        s_x[:,i-x_pixel+20] = np.abs(f_x[:,i-x_pixel+20])
    return col_x, s_x


#col_x = np.zeros(shape=(6950,100))
#for i in range(460,560):
#    col_x[:,i-460] = img[:,i]
    
col_50,  s_50   = col_fft_abs(img,1530)
col_75,  s_75   = col_fft_abs(img,1940)
col_100, s_100  = col_fft_abs(img,2330)
col_50i, s_50i  = col_fft_abs(img,2760)
col_75i, s_75i  = col_fft_abs(img,3160)
col_100i,s_100i = col_fft_abs(img,3550)
'''
col_45k, s_45k  = col_fft_abs(img,510)
col_15c, s_15c  = col_fft_abs(img,780)
col_30m, s_30m  = col_fft_abs(img,1040)
col_90y, s_90y  = col_fft_abs(img,1290)
col_50,  s_50   = col_fft_abs(img,1530)
col_75,  s_75   = col_fft_abs(img,1940)
col_100, s_100  = col_fft_abs(img,2330)
col_50i, s_50i  = col_fft_abs(img,2760)
col_75i, s_75i  = col_fft_abs(img,3160)
col_100i,s_100i = col_fft_abs(img,3550)
col_r,   s_r    = col_fft_abs(img,3950)
col_g,   s_g    = col_fft_abs(img,4150)
col_b,   s_b    = col_fft_abs(img,4345)
col_blk, s_blk  = col_fft_abs(img,4540)
#a = col[:,0]
#b = col[:,1]
#c = col[:,2]
#col_3 = (a.astype(int)+b.astype(int)+c.astype(int))/3
'''
'''
f_50 = np.fft.fft(col_50[:7000])
f_75 = np.fft.fft(col_75[:7000])
f_100 = np.fft.fft(col_100[:7000])
f_150 = np.fft.fft(col_150[:7000])
#f = np.fft.fft(aList[:4096])
print('sub after fft')
#fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
#s1 = np.log(np.abs(f))
s_50 = np.abs(f_50)
s_75 = np.abs(f_75)
s_100 = np.abs(f_100)
s_150 = np.abs(f_150)
print('calculate amplitude')
#s2 = np.log(np.abs(fshift))
'''
'''
########## calculate delta freqency ###################
sam = 6950
step = 0.0423
sam_rate = 1/step
dist = sam_rate/sam
del_f = []
for i in range(0,rows_h):
    dist_list = dist*i
    del_f.append(dist_list)
########## calculate delta freqency ###################
def plot_v(subnum,value,picname,xname):
    plt.subplot(subnum)
    plt.plot(value)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("{x}_{y} lppi".format(x=picname,y=xname),fontsize=60)
    plt.ylabel('value',fontsize=40)
'''
'''
plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_v(411,col_45k,test,'col_45k')
plot_v(412,col_15c,test,'col_15c')
plot_v(413,col_30m,test,'col_30m')
plot_v(414,col_90y,test,'col_90y')
plt.ylim((0, 300))
plt.savefig("image value_45k_15c_30m_90y_{x}.png".format(x=test))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_v(411,col_50,test,'col_50')
plot_v(412,col_75,test,'col_75')
plot_v(413,col_100,test,'col_100')
plot_v(414,col_150,test,'col_150')
plt.ylim((0, 300))
plt.savefig("image value_50_75_100_150_{x}.png".format(x=test))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_v(211,col_75i,test,'col_75i')
plot_v(212,col_100i,test,'col_100i')
plt.ylim((0, 300))
plt.savefig("image value_75i_100i_{x}.png".format(x=test))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_v(411,col_r,test,'col_r')
plot_v(412,col_g,test,'col_g')
plot_v(413,col_b,test,'col_b')
plot_v(414,col_blk,test,'col_blk')
plt.ylim((0, 300))
plt.savefig("image value_r_g_b_blk_{x}.png".format(x=test))
'''
###################################################################
'''
plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plt.subplot(411)
plt.plot(col_50) # MFP_good_1 (4000)
#plt.plot(col[:7000])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title("{x}_50 lppi".format(x=test),fontsize=60) #000MFP_brokengear_1 - MFP_good_1
plt.ylabel('value',fontsize=40)
'''
'''
#################################################################################
def plot_f(subnum,y,value,picname,xname):
    plt.subplot(subnum)
    plt.plot(y,value)
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
#    max_indx=np.argmax(ran)#max value index    
#    plt.plot(max_indx,ran[max_indx],'ks')
#    show_max='['+str(max_indx)+' '+str(ran[max_indx])+']'
#    plt.annotate(show_max,xytext=(max_indx,ran[max_indx]),xy=(max_indx,ran[max_indx]))
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("FFT_{x}_{y} lppi".format(x=picname,y=xname),fontsize=60)
    plt.ylabel('abs',fontsize=40)
    my_x_ticks = np.arange(0, 12, 0.5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 12000))
############ https://blog.csdn.net/Running_J/article/details/52119336 ####################

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_f(411,del_f,s_45k[:rows_h],test,'s_45k')
plot_f(412,del_f,s_15c[:rows_h],test,'s_15c')
plot_f(413,del_f,s_30m[:rows_h],test,'s_30m')
plot_f(414,del_f,s_90y[:rows_h],test,'s_90y')
plt.savefig("FFT_45k_15c_30m_90y_{x}.png".format(x=test))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_f(311,del_f,s_50[:rows_h],test,'s_50')
plot_f(312,del_f,s_75[:rows_h],test,'s_75')
plot_f(313,del_f,s_100[:rows_h],test,'s_100')
plt.savefig("FFT_50_75_100_150_{x}.png".format(x=test))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_f(311,del_f,s_50i[:rows_h],test,'s_50i')
plot_f(312,del_f,s_75i[:rows_h],test,'s_75i')
plot_f(313,del_f,s_100i[:rows_h],test,'s_100i')
plt.savefig("FFT_75i_100i_{x}.png".format(x=test))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_f(411,del_f,s_r[:rows_h],test,'s_r')
plot_f(412,del_f,s_g[:rows_h],test,'s_g')
plot_f(413,del_f,s_b[:rows_h],test,'s_b')
plot_f(414,del_f,s_blk[:rows_h],test,'s_blk')
plt.savefig("FFT_r_g_b_blk_{x}.png".format(x=test))

plt.show()
'''

'''
plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plt.subplot(411)
plt.plot(del_f,s_50)
#plt.plot(s1[:, 1200]) #MFP_good_MTF_600dpi-color_1(1192)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title("FFT_{x}_50 lppi".format(x=test),fontsize=60)
plt.ylabel('abs',fontsize=60)
my_x_ticks = np.arange(0, 24, 0.5)
plt.xticks(my_x_ticks)
plt.ylim((0, 12000))
'''





