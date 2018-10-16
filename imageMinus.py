# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:42:28 2018

@author: sue
"""
#https://blog.csdn.net/linczone/article/details/48414689
import cv2
import numpy as np
ph1 = "C:/Users/sue/Desktop/tt1.png"
ph2 = "C:/Users/sue/Desktop/tt2.png"
'''
img = cv2.imread(ph1)
size = img.shape
print (size)
'''

threshod= 20  #阈值

s1 = cv2.imread(ph1) #cv2.IMREAD_GRAYSCALE = 0
s2 = cv2.imread(ph2)

sub = s1 - s2

emptyimg = np.zeros(s1.shape,np.uint8)
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
'''
#pic_sub(emptyimg,s1,s2)

cv2.namedWindow("s1")
cv2.namedWindow("s2")
cv2.namedWindow("result")

cv2.imshow("s1",s1)
cv2.imshow("s2",s2)
cv2.imshow("result",sub)

cv2.waitKey(0)
cv2.destroyAllWindows()


