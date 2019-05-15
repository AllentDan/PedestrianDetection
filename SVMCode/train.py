# -*- coding: utf-8 -*-
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from os.path import dirname, join, basename
import sys
from glob import glob


bin_n = 16*16 # Number of bins

def hog(img):
    x_pixel,y_pixel=194,259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))   
    x_pixel_int=int(x_pixel/2)
    y_pixel_int=int(y_pixel/2)
    bin_cells = bins[:x_pixel_int,:y_pixel_int], bins[x_pixel_int:,:y_pixel_int], bins[:x_pixel_int,y_pixel_int:], bins[x_pixel_int:,y_pixel_int:]
    mag_cells = mag[:x_pixel_int,:y_pixel_int], mag[x_pixel_int:,:y_pixel_int], mag[:x_pixel_int,y_pixel_int:], mag[x_pixel_int:,y_pixel_int:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

img={}
num=0
for fn in glob(join(dirname(__file__)+'/posi_resized', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
#    print img[num].shape
    num=num+1
positive=num
for fn in glob(join(dirname(__file__)+'/nega_resized', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
#    print img[num].shape
    num=num+1

trainpic=[]
for i in img:
#    print type(i)
    trainpic.append(img[i])

#hogdata = [map(hog,img[i]) for i in img]
hogdata=list(map(hog,trainpic))
trainData = np.float32(hogdata).reshape(-1,bin_n*4)
responses = np.int32(np.repeat(1.0,trainData.shape[0])[:,np.newaxis])
responses[positive:trainData.shape[0]]=-1.0
#print (type(trainData))


svm = cv2.ml.SVM_create() #创建SVM model
#属性设置
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
#训练
result = svm.train(trainData,cv2.ml.ROW_SAMPLE,responses)
svm.save('svm_cat_data.dat')
######测试
test_temp=[]
for fn in glob(join(dirname(__file__)+'/predict_resized', '*.jpg')):
    img=cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
    test_temp.append(img)
testdata=list(map(hog,test_temp))
testData = np.float32(testdata).reshape(-1,bin_n*4)
re=svm.predict(testData)
