# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:56:02 2018

@author: sue
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
################# FFT ########################https://blog.csdn.net/on2way/article/details/46981825 
img = cv2.imread('sub_defect_good.bmp') #除3使用
#img = cv2.imread('sub_defect_good.bmp',0) #直接读为灰度图像
#cv2.imwrite('img.bmp', img)

col = img[:,1200]#1190為I
a = col[:,0]
b = col[:,1]
c = col[:,2]
col_gray = (a.astype(int)+b.astype(int)+c.astype(int))/3

f = np.fft.fft(col_gray)
print('sub after fft')
#fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
#s1 = np.log(np.abs(f))
s1 = np.abs(f)
print('calculate amplitude')
#s2 = np.log(np.abs(fshift))

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plt.subplot(211)
#plt.plot(img[:, 1200]) # MFP_good_1 (4000)
plt.plot(col_gray)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("_defect_good ",fontsize=60) #000MFP_brokengear_1 - MFP_good_1
plt.xlabel('pixel',fontsize=60)
plt.ylabel('value',fontsize=60)
plt.ylim((0, 300))

plt.subplot(212)
plt.plot(s1)
#plt.plot(s1[:, 1200]) #MFP_good_MTF_600dpi-color_1(1192)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("sub after FFT",fontsize=60)
plt.xlabel('pixel',fontsize=60)
plt.ylabel('abs',fontsize=60)
plt.ylim((0, 20000))

plt.savefig("fft_sub_defect_good.png")
#plt.show()

sam = 7000
step = 0.0423
sam_rate = 1/step
dist = sam_rate/sam





