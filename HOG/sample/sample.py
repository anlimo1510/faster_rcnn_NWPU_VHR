#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:02:59 2019

@author: anlimo1510
"""
#from skimage.feature import hog
#import numpy as np
import os
from PIL import Image

path = '/home/anlimo1510/projects/faster-rcnn-NWPU_VHR/data/NWPU_VHR'

# load data & label as images, labels
# path = NWPU_VHR
# read image path from train.txt

imgfile_index_path = os.path.join(path, 'image_set/train.txt')
print(imgfile_index_path)
img_indexs = []
with open(imgfile_index_path) as fimg:   
    for number in fimg.read().splitlines():
        img_indexs.append(number)
fimg.close()
# read axis from ground_truth according to every index in train.txt
save_index = 1
i = 0
for img_index in img_indexs:
    axis_path = os.path.join(path, 'ground_truth/{}.txt'.format(img_index))
    img_path = os.path.join(path, 'positive_image_set/{}.jpg'.format(img_index))
    img = Image.open(img_path)
    with open(axis_path) as f:
        for line in f.readlines():
            lineArray = line.strip().split(',')
            xmin_s = lineArray[0].strip('(').strip(')').replace(' ','')
            ymin_s = lineArray[1].strip('(').strip(')').replace(' ','')
            xmax_s = lineArray[2].strip('(').strip(')').replace(' ','')
            ymax_s = lineArray[3].strip('(').strip(')').replace(' ','')
            xmin = int(xmin_s)
            ymin = int(ymin_s)
            xmax = int(xmax_s)
            ymax = int(ymax_s)
            assert xmax >= xmin
            assert ymax >= ymin
            #cropped = img.crop((xmin,ymin,xmax,ymax))
            width = int(xmax - xmin)
            height = int(ymax - ymin)
            #save_path = os.path.join(path, 'sample_img/{}.jpg'.format(save_index))
            #cropped.save(save_path)
            label = int(lineArray[4])
            sf = open('sample.txt','a+')
            sf.writelines([str(save_index)," ",str(label)," ",str(xmin)," ",str(ymin)," ",str(xmax)," ",str(ymax)," ",str(width)," ",str(height),'\n'])
            sf.close()
            save_index += 1
    f.close()