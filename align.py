# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:20:09 2018

@author: sue
"""

##https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
import cv2
import numpy as np
import matplotlib.pyplot as plt
#cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)# Create window with freedom of dimensions
#cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Aligned Image 2", cv2.WINDOW_NORMAL)
#cv2.namedWindow("ima", cv2.WINDOW_NORMAL)
#im = cv2.imread("C:/Users/sue/Desktop/5656aa.bmp", 0)
#ret1,th1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY) #https://blog.csdn.net/on2way/article/details/46812121

#cv2.imshow('th1',th1)
#plt.imshow(th1,'gray')
#plt.axis('off')#關閉座標
#plt.savefig("th1_BINARY_127.png")

# Read the images to be aligned
img1 = 'MFP_g1'         #MFP_good_1  #MFP_good_MTF_600dpi-color_1        #_good_600dpi_test 4000*7000 3800~3950
img2 = 'MFP_g2'   #MFP_brokengear_1  #MFP_defect_MTF_600dpi-color_1  _defect_600dpi_test 4500~4640
im1 =  cv2.imread("C:/Users/User/.spyder-py3/{img1}.bmp".format(img1=img1))
im2 =  cv2.imread("C:/Users/User/.spyder-py3/{img2}.bmp".format(img2=img2))
ret1,im1_gray = cv2.threshold(im1,147,255,cv2.THRESH_BINARY)
ret2,im2_gray = cv2.threshold(im2,147,255,cv2.THRESH_BINARY)
print('loading im1: {img1}'.format(img1=img1))
print('        im2: {img2}'.format(img2=img2))
################### 橫向掃描 #############################################################################
def scan(x_high, x_low, y_left, y_right, im):
    for y in range(y_left, y_right):
        for x in range(x_high, x_low):
            if (im[x,y,0].astype(np.int16) == 0):
                return x,y
print('##################################### scan left mark ')
x1_left,y1_left = scan(890, 980, 130, 300, im1_gray) #抓im1左邊標記座標   500, 1000, 50, 300,
x2_left,y2_left = scan(890, 980, 130, 300, im2_gray) #抓im2左邊標記座標
print('im1_gray_left: [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 0])
print('im2_gray_left: [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 0])
################### 橫向掃描 #############################################################################

######### 1st平移 #################https://blog.csdn.net/on2way/article/details/46801063
print('##################################### move im2 ')
bios1 = 0
H1 = np.float32([[1,0, y1_left - y2_left + bios1],
                 [0,1, x1_left - x2_left + bios1]])
rows,cols = im2.shape[:2]
im2_mov_gray = cv2.warpAffine(im2_gray,H1,(cols,rows)) #需要图像、变换矩阵、变换后的大小
x2_mov_left,y2_mov_left = scan(890, 980, 130, 300, im2_mov_gray)  #確認結果
print('move: [{x}, {y}]'.format(x=x2_left - x1_left, y=y2_left - y1_left, b=bios1))
print('im2_mov_gray_left: [{x},{y}]'.format(x=x2_mov_left, y=y2_mov_left), im2_mov_gray[x2_mov_left, y2_mov_left, 0])
######### 1st平移 ######################################################################

######### 旋轉角度 ############################################################################################
print('##################################### scan right mark ')
x1_right,y1_right = scan(890, 980, 4591, 4640, im1_gray)     #抓im1右邊標記座標  500, 1000, 3800, 3950,
x2_mov_right,y2_mov_right = scan(890, 980, 4591, 4640, im2_mov_gray) #抓平移後im2右邊標記座標
print('im1_gray_right    : [{x1},{y1}]'.format(x1=x1_right,     y1=y1_right),     im1_gray[x1_right, y1_right, 0])
print('im2_mov_gray_right: [{x2},{y2}]'.format(x2=x2_mov_right, y2=y2_mov_right), im2_mov_gray[x2_mov_right, y2_mov_right, 0])
def angle(x1,x2,y1,y2):
    x = np.array([x1,x2])
    y = np.array([y1,y2])
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx*Ly)
    angle = np.arccos(cos_angle)
    angle_deg = angle*360/2/np.pi
    return angle_deg
print('##################################### find rotate angle ')
an = angle(x1_right - x1_left,
           y1_right - y1_left,
           x2_mov_right - x1_left,
           y2_mov_right - y1_left)
print('angle: ', an)
print('rotate')
######### 旋轉 #############################################################
#第一个参数旋转中心，第二个参数旋转角度(逆時針為正)，第三个参数：缩放比例
M = cv2.getRotationMatrix2D((x1_left,y1_left), an, 1)#以平移對齊點為中心旋轉
#im2_mov = cv2.warpAffine(im2,H1,(cols,rows))#第三个参数：变换后的图像大小
im2_mov_rot_gray = cv2.warpAffine(im2_mov_gray,M,(cols,rows))
#im2_mov_rot = cv2.warpAffine(im2_mov,M,(cols,rows))
_,im2_mov_rot_gray2 = cv2.threshold(im2_mov_rot_gray,127,255,cv2.THRESH_BINARY)
###########################################################################
#cv2.imwrite('im2_mov_rot.bmp', im2_mov_rot)

############## test zone ##############
#_,im2_mov_rot_gray = cv2.threshold(im2_mov_rot,127,255,cv2.THRESH_BINARY)

x_m_r_g2_left,y_m_r_g2_left = scan(890, 980, 130, 300, im2_mov_rot_gray2)
print('im1_gray_left      : [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 0])
print('im2_gray_left      : [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 0])
print('im2_mov_rot_gray2_l : [{x3},{y3}]'.format(x3=x_m_r_g2_left, y3=y_m_r_g2_left), 
      im2_mov_rot_gray2[x_m_r_g2_left, y_m_r_g2_left, 0])

x_m_r_g2_right,y_m_r_g2_right = scan(890, 980, 4591, 4640, im2_mov_rot_gray2)
print('im1_gray_right     : [{x1},{y1}]'.format(x1=x1_right,     y1=y1_right),     im1_gray[x1_right, y1_right, 0])
print('im2_mov_gray_right : [{x2},{y2}]'.format(x2=x2_mov_right, y2=y2_mov_right), im2_mov_gray[x2_mov_right, y2_mov_right, 0])
print('im2_mov_rot_gray2_r : [{x3},{y3}]'.format(x3=x_m_r_g2_right, y3=y_m_r_g2_right),
      im2_mov_rot_gray2[x_m_r_g2_right, y_m_r_g2_right, 0])
############## test zone ##############
######### 2nd平移 ######################################################################
print('##################################### move im2 part2 ')
bios2 = 0
H2 = np.float32([[1,0, y1_left - y_m_r_g2_left + bios2],
                 [0,1, x1_left - x_m_r_g2_left + bios2]])
im2_mov_gray2 = cv2.warpAffine(im2_mov_rot_gray2,H2,(cols,rows)) #需要图像、变换矩阵、变换后的大小
x2_mov_left2,y2_mov_left2 = scan(890, 980, 130, 300, im2_mov_gray2)
print('move: [{x}, {y}]'.format(x=x_m_r_g2_left-x1_left, y=y_m_r_g2_left-y1_left, b=bios2))
print('im2_mov_gray_left : [{x2},{y2}]'.format(x2=x2_mov_left2,  y2=y2_mov_left2),  im2_mov_gray2[x2_mov_left2, y2_mov_left2, 0])
x2_mov_right2,y2_mov_right2 = scan(890, 980, 4591, 4640, im2_mov_gray2)
print('im2_mov_gray_right: [{x2},{y2}]'.format(x2=x2_mov_right2, y2=y2_mov_right2), im2_mov_gray2[x2_mov_right2, y2_mov_right2, 0])
######### 2nd平移 ######################################################################
######### im2原圖平移>>旋轉>>平移 ###############################
print('##################################### im2_mov_rot_mov ')
im2_mov1 = cv2.warpAffine(im2,H1,(cols,rows))
im2_mov_rot = cv2.warpAffine(im2_mov1,M,(cols,rows))
im2_aligned = cv2.warpAffine(im2_mov_rot,H2,(cols,rows))
######### 影像處理 #############################################
print('im2_aligned - im1 with image process')
zero = np.zeros(im1.shape,np.uint8)
def imageMinus(res, im1, im2):
    for y in range(0, 4650):
        for x in range(0, 7000):
            for z in range(0,2):
                if (im2[x,y,z] > im1[x,y,z]):
                    res[x,y,z] = im2[x,y,z] - im1[x,y,z]
                else:
                    res[x,y,z] = 0
imageMinus(zero,im1,im2_aligned)
cv2.imwrite('{x}_{y}.bmp'.format(x=img1,y=img2), zero)



















