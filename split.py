# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:30:09 2019

@author: x

将视频分解为图片
decompose the each video into images by hand !!!!!!


calib.mp4 -> ./calib_jpg/0001.jpg
             ./calib_jpg/0002.jpg
             ...


The original image size is too large, and there is no need 
to write them in folder.
The size of the input image required by the network is 416 x 128.

1.mp4 -> ./dataset/video1/000005.jpg
         ./dataset/video1/000010.jpg
         ...

2.mp4 -> ./dataset/video2/000005.jpg
         ./dataset/video2/000010.jpg
         ...

...

"""

import numpy as np
import cv2

# cap = cv2.VideoCapture("./video/calib.mp4")
cap = cv2.VideoCapture("./video/2.mp4")


i = 0
while True:
    i += 1
    ret, frame = cap.read()
    if ret is False:
        break
    
    cv2.imshow("image", frame)
    cv2.waitKey(10)
    
    # if i%20 == 0:  # calib.mp4
    #     print("i:", i)
    #     cv2.imwrite("./calib_jpg/"  + str('%04d'%i) + ".jpg", frame)

    if i%5 == 0:
        print("i:", i)
        resized_frame = cv2.resize(frame, (416, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite("./dataset/video2/"  + str('%04d'%i) + ".jpg", resized_frame)
        
        
cap.release()
cv2.destroyAllWindows()