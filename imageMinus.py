# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:42:28 2018

@author: sue
"""
#https://blog.csdn.net/linczone/article/details/48414689
import cv2
import numpy as np
#import imutils
#from matplotlib import pyplot as plt

#from __future__ import print_function
'''
#尋找特徵及匹配 https://blog.csdn.net/yuanlulu/article/details/82222119
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches_10000_015MFP_good_1.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h
'''

#cv2.namedWindow("s1", cv2.WINDOW_NORMAL)# Create window with freedom of dimensions
#cv2.namedWindow("s2", cv2.WINDOW_NORMAL)
#cv2.namedWindow("result", cv2.WINDOW_NORMAL)

ph1 = "C:/Users/sue/Desktop/MFP_good_1.bmp"#MFP_good_1.bmp
ph2 = "C:/Users/sue/Desktop/MFP_brokengear_1.bmp"#MFP_brokengear_1.bmp
'''
img = cv2.imread(ph1)
size = img.shape
print (size)
'''
s1 = cv2.imread(ph1) #cv2.IMREAD_GRAYSCALE = 0
s2 = cv2.imread(ph2)

'''
threshold= 1  #門檻值
# 產生等高線
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 建立除錯用影像
img_debug = s1.copy()

# 線條寬度
line_width = int(s1.shape[1]/100)

# 以藍色線條畫出所有的等高線
cv2.drawContours(img_debug, contours, -1, (255, 0, 0), line_width)

# 找出面積最大的等高線區域
c = max(contours, key = cv2.contourArea)

# 找出可以包住面積最大等高線區域的方框，並以綠色線條畫出來
x, y, w, h = cv2.boundingRect(c)
cv2.rectangle(img_debug,(x, y), (x + w, y + h), (0, 255, 0), line_width)

# 嘗試在各種角度，以最小的方框包住面積最大的等高線區域，以紅色線條標示
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img_debug, [box], 0, (0, 0, 255), line_width)

# 除錯用的圖形
plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
plt.show()
'''

#img_small = imutils.resize(s1, width=2325, height=3500)
#img_small = imutils.resize(s2, width=300, height=300)
#ss1 = cv2.resize(s1, (700, 465))# Resize window to specified dimensions
#ss2 = cv2.resize(s2, (700, 465))
'''
print("Aligning images ...")
# Registered image will be resotred in imReg. 
# The estimated homography will be stored in h. 
imReg, h = alignImages(s1, s2)

# Write aligned image to disk. 
outFilename = "aligned_10000_015MFP_good_1.jpg"
print("Saving aligned image : ", outFilename); 
cv2.imwrite(outFilename, imReg)

# Print estimated homography
print("Estimated homography : \n",  h)
'''
##########################掃描對齊圖示########################
'''
for i in range(910, 916):
    for j in range(267, 270):
        if (s1[i,j] < 160):
            print('s1: ', i, j, s1[i,j])
print('done s1')

for p in range(910, 915):
    for q in range(270, 272):
        if (src[p,q] < 160):
            print('src: ', p, q, src[p,q])
'''            
img = cv2.imread(ph2)
rows,cols ,s= img.shape
 
M = np.float32([[1,0,-2],[0,1,3]])
dst = cv2.warpAffine(img,M,(4650,7000))

sub = dst - s1

#cv2.imshow('img',dst)
#cv2.waitKey(0)
cv2.imwrite('outFile.bmp', sub)
#cv2.destroyAllWindows()

##########################掃描對齊圖示########################

'''
sub = s1 - s2
#emptyimg = np.zeros(s1.shape,np.uint8)
'''
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
'''
cv2.imshow("result",sub)
cv2.waitKey(0)#键盘绑定函数, 0為輸入任意鍵後執行
cv2.destroyAllWindows()#關閉打開的視窗
'''
'''
plt.imshow(s1,  interpolation='None')
plt.show()
plt.imshow(s2,  interpolation='None')
plt.show()
plt.imshow(sub,  interpolation='None')
plt.show()
'''


