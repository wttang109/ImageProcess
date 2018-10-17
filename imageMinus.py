# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:42:28 2018

@author: sue
"""
#https://blog.csdn.net/linczone/article/details/48414689
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

#cv2.namedWindow("s1", cv2.WINDOW_NORMAL)# Create window with freedom of dimensions
#cv2.namedWindow("s2", cv2.WINDOW_NORMAL)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)

ph1 = "C:/Users/sue/Desktop/tt1.png"
ph2 = "C:/Users/sue/Desktop/tt2.png"
'''
img = cv2.imread(ph1)
size = img.shape
print (size)
'''
#threshold= 20  #門檻值
s1 = cv2.imread(ph1,0) #cv2.IMREAD_GRAYSCALE = 0
s2 = cv2.imread(ph2,0)


#img_small = imutils.resize(s1, width=2325, height=3500)
#img_small = imutils.resize(s2, width=300, height=300)

#ss1 = cv2.resize(s1, (700, 465))# Resize window to specified dimensions
#ss2 = cv2.resize(s2, (700, 465))

sub = s1 - s2

#emptyimg = np.zeros(s1.shape,np.uint8)
'''
def pic_sub(dest,s1,s2):
    for x in range(dest.shape[0]):
        for y in range(dest.shape[1]):
            if(s2[x,y] > s1[x,y]):
                dest[x,y] = s2[x,y] - s1[x,y]
            else:
                dest[x,y] = s1[x,y] - s2[x,y]

            if(dest[x,y] < threshod):
                dest[x,y] = 0
            else:
                dest[x,y] = 255

#pic_sub(emptyimg,s1,s2)
'''
#cv2.resizeWindow("s1", 500, 500)
#cv2.imshow("s1",s1)
#cv2.imshow("s2",s2)
cv2.imshow("result",sub)

cv2.waitKey(0)#键盘绑定函数, 0為輸入任意鍵後執行
cv2.destroyAllWindows()#關閉打開的視窗
'''
plt.imshow(s1,  interpolation='None')
plt.show()
plt.imshow(s2,  interpolation='None')
plt.show()
plt.imshow(sub,  interpolation='None')
plt.show()
'''



