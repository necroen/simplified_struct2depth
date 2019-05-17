# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:33:52 2019

@author: x
"""

import cv2
import numpy as np
import glob

#import matplotlib.pyplot as plt
#import matplotlib.patches as patches


# 找棋盘格角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值

#棋盘格模板规格
w = 9   # 10 - 1
h = 6   # 7  - 1

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*25  # 18.1 mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('./calib_jpg/*.jpg')  #   拍摄的十几张棋盘图片所在目录

i = 1
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i+1
    
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 360)   # 
        cv2.imshow('findCorners',img)
        cv2.waitKey(100)
        
cv2.destroyAllWindows()
#%% 标定
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("ret:", ret)
print("mtx:\n", mtx)      # 内参数矩阵  
print("dist:\n", dist)   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)  
#print("rvecs:\n",rvecs)   # 旋转向量  # 外参数  
#print("tvecs:\n",tvecs  )  # 平移向量  # 外参数

# mtx = np.array([
#  [1.14183754e+03, 0.00000000e+00, 6.28283670e+02],
#  [0.00000000e+00, 1.13869492e+03, 3.56277189e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
#  ])

    
# dist = np.array([ 8.73438827e-02,-4.44308237e-01,-2.84493359e-04,
#                  -6.76058846e-04,-3.27616998e-01])