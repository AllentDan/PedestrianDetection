# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:06:37 2019

@author: Allent_Computer
"""


from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from os.path import dirname, join, basename
import shutil
import sys
from glob import glob


bin_n = 16*16 # hog中用的

#通过指定的因子来调整图像的大小
def frame_resize(img, scaleFactor):
  return cv2.resize(img, (int(img.shape[1] * (1 / scaleFactor)),
    int(img.shape[0] * (1 / scaleFactor))), interpolation=cv2.INTER_AREA)

#建立图像金字塔，返回被调整过大小的图像直到宽度和高度都达到所规定的最小约束
def pyramid(image, scale=1.5, minSize=(100, 80)):
    # 迭代获取图像
    yield image

    while True:
        #print(image.shape)
        image = frame_resize(image, scale)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            #print(image.shape)
            break
        yield image

#非极大值抑制，消除重叠窗口
def py_nms(boxs,thresh_l=0, thresh_r=0.9, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    dets=np.array(boxs)
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        inds = np.where(ovr<=thresh_r)[0]
#        print((ovr<=thresh_r))
#        print((ovr>=thresh_l))
#        print((ovr<=thresh_r) & (ovr>=thresh_l))
        order = order[inds + 1]
    return dets[keep]

#提取图片的hog特征
def hog(img):
    x_pixel,y_pixel=352,288
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

#滑动窗口，产生图片用于预测
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
#获取训练样本，输入到trainData
img={}
num=0
for fn in glob(join(dirname(__file__)+'/posi_resized', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
    num=num+1
positive=num
for fn in glob(join(dirname(__file__)+'/nega_resized', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
    num=num+1
trainpic=[]
for i in img:
    trainpic.append(img[i])
hogdata=list(map(hog,trainpic))
trainData = np.float32(hogdata).reshape(-1,bin_n*4)
responses = np.int32(np.repeat(1.0,trainData.shape[0]))
responses[positive:trainData.shape[0]]=-1.0
trainData=list(trainData)

#获取测试样本，用于测试SVM分类效果
test_temp=[]
for fn in glob(join(dirname(__file__)+'/predict_resized', '*.jpg')):
    img=cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
    test_temp.append(img)
testdata=list(map(hog,test_temp))
testData = np.float32(testdata).reshape(-1,bin_n*4)

###该部分为训练过程代码，因为已经保存好m文件，所以注释掉了，直接load训练好的到model
#model = svm.SVC(C=0.01,gamma=1,kernel='linear',probability=True)  # class 
#, param_grid={"C":[0.001, 0.01, 0.1], "gamma": [1, 0.1, 0.01]}, cv=4
#model = GridSearchCV(svm.SVC(probability=True), param_grid={"C":[0.00001, 0.0001], "gamma": [0.5, 5]}, cv=4)

#model.fit(trainData, responses)  # training the svc model
#print("The best parameters are %s with a score of %0.2f"
#      % (model.best_params_, model.best_score_))
#joblib.dump(model, "train_model.m")
#result1 = model.predict(testData)
#result2=model.predict_proba(testData)
#print(result1)
#print(result2)

#load模型到model，并测试
model=joblib.load("E:\\Python\\SVM4Pedestrian\\train_model.m")
image=cv2.imread("E:\\Python\\detected_image.jpg")#彩色读入，用于显示
imggray=cv2.imread("E:\\Python\\detected_image.jpg",0)#灰色读入，用于预测
(winW, winH) = (64, 64)#滑动窗口大小
winWBox=int(winW)#存储高斯金字塔缩小的尺寸对应的box应该有多大
winHBox=int(winH)
boxs=[]#存储box，用于非极大抑制

for img in pyramid(imggray,1.5):#用滑动窗口+非极大抑制数人头
    for (x, y, window) in sliding_window(img, stepSize=21, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
                continue
        clone = img.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cut_img=img[y:y + winH,x:x+winW]
        cut_img=cv2.resize(cut_img,(64,128),interpolation=cv2.INTER_AREA)
        data=hog(cut_img)[None,:]
        data = np.float32(data).reshape(-1,bin_n*4)
        prediction=model.predict_proba(data)[0]
        if prediction[1] >0.98:
            scaleBox=float(winWBox)/float(winW)
            box=[x*scaleBox,y*scaleBox,x*scaleBox + winWBox,y*scaleBox + winHBox,prediction[1]]
            boxs.append(box)
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
    winWBox=winWBox*1.5
    winHBox=winHBox*1.5

#将结果画图    
result=py_nms(boxs,thresh_l=0.0,thresh_r=0.1,mode="Union")
saveNum=0
for i in range(len(result)):
    x1=int(result[i][0])
    y1=int(result[i][1])
    x2=int(result[i][2])
    y2=int(result[i][3])
    save_img=image[y1:y2,x1:x2]
    cv2.imwrite(str(saveNum)+".jpg",save_img)
    saveNum+=1
    cv2.rectangle(image, (x1,y1 ), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image,str(saveNum),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
    
cv2.putText(image,"All:"+str(saveNum),(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
    
cv2.imshow("result:",image)
k = cv2.waitKey(0) # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
if k ==27:     # 键盘上Esc键的键值
    cv2.destroyAllWindows()
cv2.imwrite("E:\\Python\\SVM4Pedestrian\\result.jpg",image)
