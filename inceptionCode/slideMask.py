# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:08 2019

@author: Allent_Computer
"""
# import the necessary packages
import helpers
import argparse
import time
import cv2


# load the image and define the window width and height
image = cv2.imread('E:/python/myInceptionProject/detected_image.jpg')  
(winW, winH) = (60, 48)

i=1
# loop over the image pyramid
for resized in helpers.pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in helpers.sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        # since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
#        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cut_img=clone[y:y + winH,x:x+winH]
        cv2.imwrite("E:/python/myInceptionProject/cut_img/"+str(i)+'.jpg',cut_img)
        i+=1
#        print(img)
#        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
#        cv2.imshow("Window", clone)
#        cv2.waitKey(1)
#    k = cv2.waitKey(0) # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
#    if k ==27:     # 键盘上Esc键的键值
#        cv2.destroyAllWindows()
#    continue
#         time.sleep(0.025)
