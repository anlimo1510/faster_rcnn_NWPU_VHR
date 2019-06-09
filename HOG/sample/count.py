#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:04:11 2019

@author: anlimo1510
"""

#import numpy as np
from scipy import stats
from collections import Counter

count_width = []
count_height = []
count_label = []
count_size = []
ff = open('sample.txt','r')
for line in ff.readlines():
    lineArray = line.strip().split(' ')
    img_index = lineArray[0]
    label = lineArray[1]
    width = int(lineArray[6])
    height = int(lineArray[7])
    count_label.append(label)
    count_width.append(width)
    count_height.append(height)
    count_size.append(width*height)
ff.close()

print(max(count_size))
index = count_size.index(max(count_size))
print(index)
print(count_width[index],' ',count_height[index])
#print(max(count_width),min(count_width))
#print(max(count_height),min(count_height))

#print(stats.mode(count_width)[0][0])
#print(stats.mode(count_height)[0][0])

#print(sum(count_width)/len(count_width))
#print(sum(count_height)/len(count_height))

print(Counter(count_label))