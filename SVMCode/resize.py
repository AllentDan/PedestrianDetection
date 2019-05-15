# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:02:15 2019

@author: Allent_Computer
"""


import numpy as np
import cv2
from os.path import dirname, join, basename
from glob import glob
 
num=0
for fn in glob(join(dirname(__file__)+'/nega','*.jpg')):    #获取位置的nega文件夹下所有的jpg图片，nega文件夹下面是我存放的所有负样本
    print(fn)
    img = cv2.imread(fn)
    res=cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)#线性插值并统一尺寸
    cv2.imwrite('E:\\Python\\SVM4Pedestrian\\nega_resized\\'+str(num)+'.jpg',res)#新建nega_resized文件夹，将所有处理后的图片写到该文件夹
    num=num+1
 
print ('all done!')