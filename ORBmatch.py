# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:58:24 2018

@author: sue
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('C:/Users/sue/Desktop/pic1.bmp',0) # queryImage
img2 = cv2.imread('C:/Users/sue/Desktop/pic2.bmp',0) # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:4],None, flags=2)

plt.imshow(img3),plt.show()

##############新增區#################
cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
cv2.imshow('img3', img3)
cv2.waitKey(0)#键盘绑定函数, 0為輸入任意鍵後執行
cv2.destroyAllWindows()#關閉打開的視窗



