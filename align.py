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
img1 = 'MFP_good_1.bmp'         #MFP_good_1  #MFP_good_MTF_600dpi-color_1
img2 = 'MFP_brokengear_1.bmp'   #MFP_brokengear_1  #MFP_defect_MTF_600dpi-color_1
im1 =  cv2.imread("C:/Users/sue/wtt/pic/{img1}".format(img1=img1))
im2 =  cv2.imread("C:/Users/sue/wtt/pic/{img2}".format(img2=img2))
ret1,im1_gray = cv2.threshold(im1,127,255,cv2.THRESH_BINARY)
ret2,im2_gray = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)
print('loading im1: {img1}'.format(img1=img1))
print('        im2: {img2}'.format(img2=img2))
################### 橫向掃描 #############################################################################
def scan(x_high, x_low, y_left, y_right, im):
    for y in range(y_left, y_right):
        for x in range(x_high, x_low):
            if (im[x,y,2].astype(np.int16) == 0):
                return x,y
print('######## scan left mark ###############')
x1_left,y1_left = scan(500, 1000, 50, 300, im1_gray) #抓im1左邊標記座標
x2_left,y2_left = scan(500, 1000, 50, 300, im2_gray) #抓im2左邊標記座標
print('im1_gray_left: [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 2])
print('im2_gray_left: [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 2])
################### 橫向掃描 #############################################################################

######### 1st平移 #################https://blog.csdn.net/on2way/article/details/46801063
print('######## move im2 #####################')
bios1=1
H1 = np.float32([[1,0, x2_left - x1_left + bios1],
                 [0,1, y2_left - y1_left + bios1]])
rows,cols = im2.shape[:2]
im2_mov_gray = cv2.warpAffine(im2_gray,H1,(cols,rows)) #需要图像、变换矩阵、变换后的大小
x2_mov_left,y2_mov_left = scan(500, 1000, 50, 300, im2_mov_gray)  #確認結果
print('move: [{x}+{b}, {y}+{b}]'.format(x=x2_left - x1_left, y=y2_left - y1_left, b = bios1))
print('im2_mov_gray_left: [{x},{y}]'.format(x=x2_mov_left, y=y2_mov_left), im2_mov_gray[x2_mov_left, y2_mov_left, 2])
######### 1st平移 ######################################################################

######### 旋轉角度 ############################################################################################
print('######## scan right mark ##############')
x1_right,y1_right = scan(500, 1000, 4500, 4640, im1_gray)     #抓im1右邊標記座標
x2_mov_right,y2_mov_right = scan(500, 1000, 4500, 4640, im2_mov_gray) #抓平移後im2右邊標記座標
print('im1_gray_right    : [{x1},{y1}]'.format(x1=x1_right, y1=y1_right), im1_gray[x1_right, y1_right, 2])
print('im2_mov_gray_right: [{x2},{y2}]'.format(x2=x2_mov_right, y2=y2_mov_right), im2_mov_gray[x2_mov_right, y2_mov_right, 2])
def angle(x1,x2,y1,y2):
    x = np.array([x1,x2])
    y = np.array([y1,y2])
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx*Ly)
    angle = np.arccos(cos_angle)
    angle_deg = angle*360/2/np.pi
    return angle_deg
print('######## find rotate angle ############')
an = angle(x1_right - x1_left,
           y1_right - y1_left,
           x2_mov_right - x1_left,
           y2_mov_right - y1_left)
print('angle: ', an)
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

x_m_r_g2_left,y_m_r_g2_left = scan(500, 1000, 50, 300, im2_mov_rot_gray2)
print('im1_gray_left      : [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 2])
print('im2_gray_left      : [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 2])
print('im2_mov_rot_gray2_l : [{x3},{y3}]'.format(x3=x_m_r_g2_left, y3=y_m_r_g2_left), 
      im2_mov_rot_gray2[x_m_r_g2_left, y_m_r_g2_left, 2])

x_m_r_g2_right,y_m_r_g2_right = scan(500, 1000, 4500, 4640, im2_mov_rot_gray2)
print('im1_gray_right     : [{x1},{y1}]'.format(x1=x1_right, y1=y1_right), im1_gray[x1_right, y1_right, 2])
print('im2_mov_gray_right : [{x2},{y2}]'.format(x2=x2_mov_right, y2=y2_mov_right), im2_mov_gray[x2_mov_right, y2_mov_right, 2])
print('im2_mov_rot_gray2_r : [{x3},{y3}]'.format(x3=x_m_r_g2_right, y3=y_m_r_g2_right),
      im2_mov_rot_gray2[x_m_r_g2_right, y_m_r_g2_right, 2])
############## test zone ##############
######### 2nd平移 ######################################################################
print('######## move im2 part2 ###############')
bios2 = -1
H2 = np.float32([[1,0, x_m_r_g2_left - x1_left + bios2],
                 [0,1, y_m_r_g2_left - y1_left + bios2]])
im2_mov_gray2 = cv2.warpAffine(im2_mov_rot_gray2,H2,(cols,rows)) #需要图像、变换矩阵、变换后的大小
x2_mov_left2,y2_mov_left2 = scan(500, 1000, 50, 300, im2_mov_gray2)
print('move: [{x}{b}, {y}{b}]'.format(x=x_m_r_g2_left-x1_left, y=y_m_r_g2_left-y1_left,b=bios2))
print('im2_mov_gray_left : [{x2},{y2}]'.format(x2=x2_mov_left2, y2=y2_mov_left2), im2_mov_gray2[x2_mov_left2, y2_mov_left2, 2])
x2_mov_right2,y2_mov_right2 = scan(500, 1000, 4500, 4640, im2_mov_gray2)
print('im2_mov_gray_right: [{x2},{y2}]'.format(x2=x2_mov_right2, y2=y2_mov_right2), im2_mov_gray2[x2_mov_right2, y2_mov_right2, 2])
######### 2nd平移 ######################################################################
######### im2原圖平移>>旋轉>>平移 ##################
print('######## im2_mov_rot_mov ##############')
im2_mov1 = cv2.warpAffine(im2,H1,(cols,rows))
im2_mov_rot = cv2.warpAffine(im2_mov1,M,(cols,rows))
im2_aligned = cv2.warpAffine(im2_mov_rot,H2,(cols,rows))


sub = im2_aligned - im1
cv2.imwrite('sub.bmp', sub)
print('im2_aligned - im1')

################# FFT ########################https://blog.csdn.net/on2way/article/details/46981825 
img = cv2.imread('sub.bmp',0) #直接读为灰度图像
f = np.fft.fft(img)
print('sub after fft')
#fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
#s1 = np.log(np.abs(f))
s1 = np.abs(f)
print('calculate amplitude')
#s2 = np.log(np.abs(fshift))
#plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original')
#plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center')
#cv2.imwrite('s1.bmp',s1)
#cv2.imwrite('s2.bmp',s2)
'''
plt.figure(figsize=(80,20),dpi=100,linewidth = 0.9)
plt.plot(s1[:,4000]) #MFP_good_MTF_600dpi-color_1(1192)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("sub after FFT",fontsize = 60)
plt.xlabel('pixel',fontsize=60)
plt.ylabel('abs',fontsize=60)
plt.savefig("fft_sub_gray.png")
#plt.show()
'''
col = sub[:,1190]#1190為I
a = col[:,0]
b = col[:,1]
c = col[:,2]
col_av = (a.astype(int)+b.astype(int)+c.astype(int))/3
#sub_gray = cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY)
f = np.fft.fft(col_av)

#fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
#s1 = np.log(np.abs(f))
s1 = np.abs(f)
#s2 = np.log(np.abs(fshift))
#plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original')
#plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center')
#cv2.imwrite('s1.bmp',s1)
#cv2.imwrite('s2.bmp',s2)

plt.figure(figsize=(100,40),dpi=100,linewidth = 0.9)
plt.subplot(211)
plt.plot(col_av[:])
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("000MFP_defect_MTF_600dpi-color_1_average ",fontsize = 60)
plt.xlabel('pixel',fontsize=60)
plt.ylabel('value',fontsize=60)
plt.ylim((0, 200))

plt.subplot(212)
plt.plot(s1[:])
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("000MFP_defect_MTF_600dpi-color_1col_av by FFT",fontsize = 60)
plt.xlabel('pixel',fontsize=60)
plt.ylabel('amplitude',fontsize=60)
plt.ylim((0, 20000))

plt.savefig("000MFP_defect_MTF_600dpi-color_1.png")
#plt.show()

'''
# Convert images to grayscale
#im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
#im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
# Find size of image1
sz = im1.shape
 
# Define the motion model
warp_mode = cv2.MOTION_TRANSLATION# MOTION_EUCLIDEAN
 
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)#對角矩陣
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
 
# Specify the number of iterations.
number_of_iterations = 500
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
 
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography 
    im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
sub = im2_aligned - im1 
#sub_gray = cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY)
#q = im2 - im1 
# Show final results
#cv2.imshow("Image 1", im1)
#cv2.imshow("Image 2", im2)
#cv2.imshow("Aligned Image 2", im2_aligned)
cv2.imwrite('Aligned Image 2.bmp', im2_aligned)
#cv2.imshow("im2_aligned_im1", sub)
cv2.imwrite('sub_BINARY_bg_500.bmp', sub)
#cv2.imwrite('q.bmp', q)
#cv2.waitKey(0)
##########################擷取部分影像#############################
x = 600
y = 200
w = 400
hh = 400
cut_sub = sub[y : y+hh, x : x+w]
#cut_s2 = s2[y : y+hh, x : x+w]
cv2.imwrite('cut_sub.bmp',cut_sub)
#cv2.imwrite('cut2.bmp',cut_s2)
##########################擷取部分影像#############################
'''