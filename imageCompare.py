# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:50:01 2018

@author: sue
"""
#https://blog.csdn.net/ibaymin/article/details/74936742
from skimage.measure import compare_ssim
#~ import skimage  as ssim
import argparse
import imutils
import cv2

#加载两张图片并将他们转换为灰度
imageA = cv2.imread("C:/Users/sue/Desktop/101.png")
imageB = cv2.imread("C:/Users/sue/Desktop/102.png")

grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

#计算两个灰度图像之间的结构相似度指数
(score,diff) = compare_ssim(grayA, grayB, full = True)
diff = (diff *255).astype("uint8")
print("SSIM:{}".format(score))

#找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#找到一系列区域，在区域周围放置矩形
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(imageA,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.rectangle(imageB,(x,y),(x+w,y+h),(0,0,255),2)

#用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
cv2.imshow("Modified",imageB)
cv2.imwrite("haha2.png",imageB)
cv2.waitKey(0)