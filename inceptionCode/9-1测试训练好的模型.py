# coding: utf-8
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import helpers
import cv2
import imutils
 
# load the image and define the window width and height
image = cv2.imread('E:/python/myInceptionProject/test_img/ped_sample1900.jpg')  
(winW, winH) = (60, 48)

lines = tf.gfile.GFile('retrain/output_labels.txt').readlines()
uid_to_human = {}
k=2

def py_nms(boxs, thresh=0.9, mode="Union"):
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
    scores = dets[:, 2]
    x2 = x1+winW
    y2 = y1+winW
    

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

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


#一行一行读取数据
for uid,line in enumerate(lines) :
    #去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line
 
def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]
 
with tf.gfile.FastGFile('retrain/output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录
scale=2.5
w = int(image.shape[1] / scale)
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录
    for root,dirs,files in os.walk('E:/python/myInceptionProject/test_img/'):
        for file in files:
            boxs=[]
            image=cv2.imread(os.path.join(root,file))
            resized = imutils.resize(image, width=w)
            #for resized in helpers.pyramid(image, scale=1.5):
                # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in helpers.sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                clone = resized.copy()
            #        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cut_img=clone[y:y + winH,x:x+winH]
                cv2.imwrite("E:/python/myInceptionProject/cut_img/1.jpg",cut_img)
                #载入图片
                image_data = tf.gfile.FastGFile("E:/python/myInceptionProject/cut_img/1.jpg", 'rb').read()
                predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
                predictions = np.squeeze(predictions)#把结果转为1维数据
                
                #排序
                top_k = predictions.argsort()[::-1]
                if top_k[0]==3 and predictions[top_k[0]]>0.92:
                    box=[x,y,predictions[top_k[0]]]
                    boxs.append(box)
#                    cv2.rectangle(image, (int_x, int_y), (int_x + int_winW, int_y + int_winW), (0, 255, 0), 1)
                    #打印图片路径及名称
            #            image_path = os.path.join(root,file)
            #            print(image_path)
                    #显示图片
            #            img=Image.open(image_path)
#                    plt.imshow(cut_img)
#                    plt.axis('off')
#                    plt.show()
#                    print(top_k)
                    for node_id in top_k:     
                        #获取分类名称
                        human_string = id_to_string(node_id)
                        #获取该分类的置信度
                        score = predictions[node_id]
                        print('%s (score = %.5f)' % (human_string, score))
            result=py_nms(boxs,thresh=0.3,mode="Union")
            for i in range(len(result)):
                int_x=int(result[i][0]*scale)
                int_y=int(result[i][1]*scale)
                int_winW=int(winW*scale)
                int_winH=int(winH*scale)
                cv2.rectangle(image, (int_x, int_y), (int_x + int_winW, int_y + int_winH), (0, 255, 0), 1)
            cv2.imwrite("E:/python/myInceptionProject/cut_img/"+str(k)+".jpg",image)
            k+=1
