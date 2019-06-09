#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:37:46 2019

@author: anlimo1510
"""
import os
#from scipy import stats
from collections import Counter

path = '/home/anlimo1510/projects/faster-rcnn-NWPU_VHR/data/NWPU_VHR'

count_width = []
count_height = []
count_label = []
count_size = []
image_index = []
img_indexs = range(1,650)
for img_index in img_indexs:
    s = '%03d'%img_index
    axis_path = os.path.join(path, 'ground_truth/{}.txt'.format(s))
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
            count_label.append(label)
            count_width.append(width)
            count_height.append(height)
            count_size.append(width*height)
            image_index.append(img_index)
f.close()

print('max size: ',max(count_size))
imgindex = image_index[count_size.index(max(count_size))]
print('max object img-index: ',imgindex)
objindex = count_size.index(max(count_size))
print('max object w*h: ',count_width[objindex],' ',count_height[objindex])

print('min size: ',min(count_size))
imgindex = image_index[count_size.index(min(count_size))]
print('min object img-index: ',imgindex)
objindex = count_size.index(min(count_size))
print('min object w*h: ',count_width[objindex],' ',count_height[objindex])
#print(max(count_width),min(count_width))
#print(max(count_height),min(count_height))

#print(stats.mode(count_width)[0][0])
#print(stats.mode(count_height)[0][0])

#print(sum(count_width)/len(count_width))
#print(sum(count_height)/len(count_height))

print('label count: ',Counter(count_label))
print('total label num: ',len(count_label))


