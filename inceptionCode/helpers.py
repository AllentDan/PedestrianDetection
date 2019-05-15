# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:43:36 2019

@author: Allent_Computer
"""

# import the necessary packages
import imutils
from skimage.transform import pyramid_gaussian
import cv2

def pyramid(image, scale=1.5, minSize=(70, 70)):
    # yield the original image
    yield image
    loop=1
    # keep looping over the pyramid
    while loop:
        loop-=1
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

if __name__ == '__main__':
    image = cv2.imread('E:/python/myInceptionProject/detected_image.jpg')  
    # METHOD #2: Resizing + Gaussian smoothing.
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=3)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        # show the resized image
        WinName = "Layer {}".format(i + 1)
#        cv2.imshow(WinName, resized)
#        cv2.waitKey(10)
        resized = resized*255
        cv2.imwrite('./'+WinName+'.jpg',resized)

