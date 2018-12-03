# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:20:09 2018

@author: sunny
"""

## https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
## http://duncancloud.blogspot.com/2016/06/python.html
import cv2
import numpy as np
import os

fileList = []
folderCount = 0
rootdir = 'eut'

for root, subFolders, files in os.walk(rootdir):
    for file in files:
        f = os.path.join(root,file)
        #print(f)
        fileList.append(f)

for i in range(1,len(fileList)):
    # Read the images to be aligned
    img1 = fileList[0]         #MFP_good_1  #MFP_good_MTF_600dpi-color_1        #_good_600dpi_test 4000*7000 3800~3950
    img2 = fileList[i]   #MFP_brokengear_1  #MFP_defect_MTF_600dpi-color_1  _defect_600dpi_test 4500~4640
    im1 =  cv2.imread(img1)
    im2 =  cv2.imread(img2)
    ret1,im1_gray = cv2.threshold(im1,147,255,cv2.THRESH_BINARY)
    ret2,im2_gray = cv2.threshold(im2,147,255,cv2.THRESH_BINARY)
    print('loaded im1: {img1}'.format(img1=img1))
    print('       im2: {img2}'.format(img2=img2))
################### scan left mark #############################################################################
    Lx_high, Lx_low, Ly_left, Ly_right=[890, 980, 130, 300]
    Rx_high, Rx_low, Ry_left, Ry_right=[917, 980, 4591, 4640]
    def scan(x_high, x_low, y_left, y_right, im):
        for y in range(y_left, y_right):
            for x in range(x_high, x_low):
                if (im[x,y,0].astype(np.int16) == 0):
                    return x,y
    print('##################################### scan left mark ')
    x1_left,y1_left = scan(Lx_high, Lx_low, Ly_left, Ly_right, im1_gray) # get im1 left mark   500, 1000, 50, 300,
    x2_left,y2_left = scan(Lx_high, Lx_low, Ly_left, Ly_right, im2_gray) # get im2 left mark
    print('im1_gray_left: [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 0])
    print('im2_gray_left: [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 0])
################### scan left mark #############################################################################

######### 1st move #################https://blog.csdn.net/on2way/article/details/46801063
    print('##################################### move im2 ')
    H1 = np.float32([[1,0, y1_left - y2_left],
                     [0,1, x1_left - x2_left]])
    rows,cols = im2.shape[:2]
    im2_mov_gray = cv2.warpAffine(im2_gray,H1,(cols,rows)) # input、transfer matrix、size
    x2_mov_left,y2_mov_left = scan(Lx_high, Lx_low, Ly_left, Ly_right, im2_mov_gray)  # confirm part
    print('move: [{x}, {y}]'.format(x=x2_left - x1_left, y=y2_left - y1_left))
    print('im2_mov_gray_left: [{x},{y}]'.format(x=x2_mov_left, y=y2_mov_left), im2_mov_gray[x2_mov_left, y2_mov_left, 0])
######### 1st move ######################################################################

######### find rotated angle ############################################################################################
    print('##################################### scan right mark ')
    x1_right,y1_right = scan(Rx_high, Rx_low, Ry_left, Ry_right, im1_gray)     # get im1 right mark  500, 1000, 3800, 3950,
    x2_mov_right,y2_mov_right = scan(Rx_high, Rx_low, Ry_left, Ry_right, im2_mov_gray) # get im2(moved) right mark
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
    print('rotating...')
######### rotate #############################################################
# rotate pivot，rotated angle(positive by counter clk)，scaling ratio
    M = cv2.getRotationMatrix2D((x1_left,y1_left), an, 1)#以平移對齊點為中心旋轉
#im2_mov = cv2.warpAffine(im2,H1,(cols,rows))
    im2_mov_rot_gray = cv2.warpAffine(im2_mov_gray,M,(cols,rows))
#im2_mov_rot = cv2.warpAffine(im2_mov,M,(cols,rows))
    _,im2_mov_rot_gray2 = cv2.threshold(im2_mov_rot_gray,127,255,cv2.THRESH_BINARY)
###########################################################################

############## check zone ##############
    x_m_r_g2_left,y_m_r_g2_left = scan(Lx_high, Lx_low, Ly_left, Ly_right, im2_mov_rot_gray2)
    print('im1_gray_left      : [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 0])
    print('im2_gray_left      : [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 0])
    print('im2_mov_rot_gray2_l : [{x3},{y3}]'.format(x3=x_m_r_g2_left, y3=y_m_r_g2_left),
          im2_mov_rot_gray2[x_m_r_g2_left, y_m_r_g2_left, 0])

    x_m_r_g2_right,y_m_r_g2_right = scan(Rx_high, Rx_low, Ry_left, Ry_right, im2_mov_rot_gray2)
    print('im1_gray_right     : [{x1},{y1}]'.format(x1=x1_right,     y1=y1_right),     im1_gray[x1_right, y1_right, 0])
    print('im2_mov_gray_right : [{x2},{y2}]'.format(x2=x2_mov_right, y2=y2_mov_right), im2_mov_gray[x2_mov_right, y2_mov_right, 0])
    print('im2_mov_rot_gray2_r : [{x3},{y3}]'.format(x3=x_m_r_g2_right, y3=y_m_r_g2_right),
          im2_mov_rot_gray2[x_m_r_g2_right, y_m_r_g2_right, 0])
############## check zone ##############
######### 2nd move ######################################################################
    print('##################################### move im2 part2 ')
    H2 = np.float32([[1,0, y1_left - y_m_r_g2_left],
                     [0,1, x1_left - x_m_r_g2_left]])
    im2_mov_gray2 = cv2.warpAffine(im2_mov_rot_gray2,H2,(cols,rows))
    x2_mov_left2,y2_mov_left2 = scan(Lx_high, Lx_low, Ly_left, Ly_right, im2_mov_gray2)
    print('move: [{x}, {y}]'.format(x=x_m_r_g2_left-x1_left, y=y_m_r_g2_left-y1_left))
    print('im2_mov_gray_left : [{x2},{y2}]'.format(x2=x2_mov_left2,  y2=y2_mov_left2),  im2_mov_gray2[x2_mov_left2, y2_mov_left2, 0])
    x2_mov_right2,y2_mov_right2 = scan(Rx_high, Rx_low, Ry_left, Ry_right, im2_mov_gray2)
    print('im2_mov_gray_right: [{x2},{y2}]'.format(x2=x2_mov_right2, y2=y2_mov_right2), im2_mov_gray2[x2_mov_right2, y2_mov_right2, 0])
######### 2nd move ######################################################################
######### im2 1st moved >> rotated >> 2nd moved ###############################
    print('##################################### im2_mov_rot_mov ')
    im2_mov1 = cv2.warpAffine(im2,H1,(cols,rows))
    im2_mov_rot = cv2.warpAffine(im2_mov1,M,(cols,rows))
    im2_aligned = cv2.warpAffine(im2_mov_rot,H2,(cols,rows))
######### process #############################################
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
    #https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/357480/
    img1_s = img1.split('\\').pop().split('/').pop().rsplit('.', 1)[0]
    img2_s = img2.split('\\').pop().split('/').pop().rsplit('.', 1)[0]

    cv2.imwrite('eut\{x}_{y}.bmp'.format(x=img1_s,y=img2_s), zero)
    print('completed {x}/{y}'.format(x=i,y=len(fileList)-1))










