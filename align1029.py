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
im1 =  cv2.imread("C:/Users/sue/Desktop/MFP_good_1.bmp")#MFP_good_1  #MFP_good_MTF_600dpi-color_1
im2 =  cv2.imread("C:/Users/sue/Desktop/MFP_brokengear_1.bmp")#MFP_brokengear_1  #MFP_defect_MTF_600dpi-color_1

ret1,im1_gray = cv2.threshold(im1,127,255,cv2.THRESH_BINARY)
ret2,im2_gray = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)

################### 橫向掃描 #############################
def scan(x_high, x_low, y_left, y_right, im):
    for y in range(y_left, y_right):
        for x in range(x_high, x_low):
            if (im[x,y,2].astype(np.int16) == 0):
                return x,y
x1_left,y1_left = scan(500, 1000, 50, 300, im1_gray)
x2_left,y2_left = scan(500, 1000, 50, 300, im2_gray)
print('im1_gray_left: [{x1},{y1}]'.format(x1=x1_left, y1=y1_left), im1_gray[x1_left, y1_left, 2])
print('im2_gray_left: [{x2},{y2}]'.format(x2=x2_left, y2=y2_left), im2_gray[x2_left, y2_left, 2])
#######################################################

#########平移######https://blog.csdn.net/on2way/article/details/46801063
H = np.float32([[1,0, x2_left-x1_left+1],[0,1, y2_left-y1_left+1]])
rows,cols = im2.shape[:2]
im2_mov_gray = cv2.warpAffine(im2_gray,H,(cols,rows)) #需要图像、变换矩阵、变换后的大小
print('move: {x}+1, {y}+1'.format(x=x2_left-x1_left, y=y2_left-y1_left))

#########旋轉角度######
x1_right,y1_right = scan(500, 1000, 4500, 4640, im1_gray)
x2_right,y2_right = scan(500, 1000, 4500, 4640, im2_mov_gray)
print('im1_gray_right: [{x1},{y1}]'.format(x1=x1_right, y1=y1_right), im1_gray[x1_right, y1_right, 2])
print('im2_mov_gray_right: [{x2},{y2}]'.format(x2=x2_right, y2=y2_right), im2_mov_gray[x2_right, y2_right, 2])
def angle(x1,x2,y1,y2):
    x = np.array([x1,x2])
    y = np.array([y1,y2])
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx*Ly)
    angle = np.arccos(cos_angle)
    return angle
an = angle(x1_right,y1_right,x2_right,y2_right)
print('angle: ', an)

#########旋轉######
#第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
M = cv2.getRotationMatrix2D((cols/2,rows/2), an, 1)
#第三个参数：变换后的图像大小
im2_mov = cv2.warpAffine(im2,H,(cols,rows))
res = cv2.warpAffine(im2_mov,M,(cols,rows))

cv2.imwrite('res_M_R.bmp', res)

_,res_gray = cv2.threshold(res,127,255,cv2.THRESH_BINARY)

xx1,yy1 = scan(500, 1000, 50, 300, im1_gray)
xx2,yy2 = scan(500, 1000, 50, 300, res_gray)
print('im1_gray: [{x1},{y1}]'.format(x1=xx1, y1=yy1), im1_gray[xx1, yy1, 2])
print('res_gray: [{x2},{y2}]'.format(x2=xx2, y2=yy2), res_gray[xx2, yy2, 2])

xx1r,yy1r = scan(500, 1000, 4500, 4640, im1_gray)
xx2r,yy2r = scan(500, 1000, 4500, 4640, res_gray)
print('im1_gray_r: [{x1},{y1}]'.format(x1=xx1r, y1=yy1r), im1_gray[xx1r, yy1r, 2])
print('res_gray_r: [{x2},{y2}]'.format(x2=xx2r, y2=yy2r), res_gray[xx2r, yy2r, 2])



#sub = im2_mov - im1
#cv2.imwrite('sub.bmp', sub)
#print('aligned im2 - im1')






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
'''
'''
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
'''
'''
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
'''
#################FFT########################https://blog.csdn.net/on2way/article/details/46981825 
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

plt.figure(figsize=(80,20),dpi=100,linewidth = 0.9)
plt.plot(s1[:,1192])#MFP_good_MTF_600dpi-color_1(1192)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("sub after FFT",fontsize = 60)
plt.xlabel('pixel',fontsize=60)
plt.ylabel('abs',fontsize=60)

plt.savefig("fft_sub_gray.png")
#plt.show()

'''














