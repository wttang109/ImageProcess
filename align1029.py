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
im1 =  cv2.imread("C:/Users/sue/Desktop/5656aa.bmp",0)#MFP_good_1
im2 =  cv2.imread("C:/Users/sue/Desktop/bbb.bmp",0)#MFP_brokengear_1

ret1,im1_gray = cv2.threshold(im1,127,255,cv2.THRESH_BINARY)
ret2,im2_gray = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)

th = 0
#########平移######https://blog.csdn.net/on2way/article/details/46801063
for i in range(0, 100):
    for j in range(0, 16):
        if (im1_gray[j,i].astype(np.int16) == 0):
            print('im1_gray: ', i, j, im1_gray[j,i])
        
            
#print('im1_gray: ', i, j, im1_gray[j,i])

print('done s1')
'''
for p in range(910, 915):
    for q in range(270, 272):
        if (im2_gray[p,q] < 160):
            print('im2_gray: ', p, q, im2_gray[p,q])
'''





#########平移######
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

#################FFT########################https://blog.csdn.net/on2way/article/details/46981825 
#img = cv2.imread('sub.bmp',0) #直接读为灰度图像
f = np.fft.fft(sub_gray)

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

plt.figure(figsize=(80,20),dpi=100,linewidth = 0.9)
plt.plot(s1[:,1163])
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title("sub by FFT",fontsize = 60)
plt.xlabel('pixel',fontsize=60)
plt.ylabel('abs',fontsize=60)

plt.savefig("fft_sub_gray.png")
#plt.show()

'''














