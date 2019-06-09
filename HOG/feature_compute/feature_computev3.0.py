#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:24:10 2019

@author: anlimo1510
"""
from skimage.feature import hog
import os
import cv2
import numpy as np

fds = []

ff = open('sample.txt','r')
for line in ff.readlines():
    lineArray = line.strip().split(' ')
    img_index = lineArray[0]
    label = lineArray[1]
    width = int(lineArray[6])
    height = int(lineArray[7])
    img_path = os.path.join('/home/anlimo1510/projects/HOG/sample/sample_img/{}.jpg'.format(img_index))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (48,60))
    fd = hog(img_resize,
             orientations=9,
             pixels_per_cell=(8,8),
             cells_per_block=(2,2),
             visualise=False)
    #plt.imshow(hog_image, cmap=plt.cm.gray)
    #save_path = os.path.join('./hog_image/{}.png'.format(img_index))
    #plt.savefig(save_path)
    #plt.show()
    fds.append(fd)
ff.close()
#plt.imshow(hog_image, cmap=plt.cm.gray)

fds_np = np.array(fds)
np.save('fds2.npy',fds_np)