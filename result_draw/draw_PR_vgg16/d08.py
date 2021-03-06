#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:05:48 2019

@author: anlimo1510
"""

import pickle
import matplotlib.pyplot as plt



fr = open('./vgg16_faster_rcnn_iter_10000/storage tank_pr.pkl','rb')
inf = pickle.load(fr)
fr.close()
rec1 = tuple(inf.values())[0]
prec1 = tuple(inf.values())[1]
ap1 = tuple(inf.values())[2]
fr.close()

fr = open('./vgg16_faster_rcnn_iter_20000/storage tank_pr.pkl','rb')
inf = pickle.load(fr)
fr.close()
rec2 = tuple(inf.values())[0]
prec2 = tuple(inf.values())[1]
ap2 = tuple(inf.values())[2]
fr.close()
#plt.rcParams['figure.figsize'] = (8,4.5)

fr = open('./vgg16_faster_rcnn_iter_25000/storage tank_pr.pkl','rb')
inf = pickle.load(fr)
fr.close()
rec3 = tuple(inf.values())[0]
prec3 = tuple(inf.values())[1]
ap3 = tuple(inf.values())[2]
fr.close()

fr = open('./vgg16_faster_rcnn_iter_30000/storage tank_pr.pkl','rb')
inf = pickle.load(fr)
fr.close()
rec4 = tuple(inf.values())[0]
prec4 = tuple(inf.values())[1]
ap4 = tuple(inf.values())[2]
fr.close()

fr = open('./vgg16_faster_rcnn_iter_40000/storage tank_pr.pkl','rb')
inf = pickle.load(fr)
fr.close()
rec5 = tuple(inf.values())[0]
prec5 = tuple(inf.values())[1]
ap5 = tuple(inf.values())[2]
fr.close()

#print(ap1,ap2,ap3,ap4,ap5)

rec1 = rec1[0:1915]
rec2 = rec2[0:1915]
plt.rcParams['savefig.dpi'] = 300 #图片像素
fig = plt.figure(num=1)
plt.plot(rec1[0:1915],prec1[0:1915],color = 'palegreen', label = "iter=10000",linewidth=0.618)
plt.plot(rec2[0:1915],prec2[0:1915],color = 'red',linestyle = '-.', label = "iter=20000",linewidth=0.618)
plt.plot(rec3[0:1915],prec3[0:1915],color = 'blue', linestyle = '-', label = "iter=25000",linewidth=0.618)
plt.plot(rec4[0:1915],prec4[0:1915],color = 'gold',linestyle = ':', label = "iter=30000",linewidth=0.618)
plt.plot(rec5[0:1915],prec5[0:1915],color = 'darkviolet',linestyle = '--', label = "iter=40000",linewidth=0.618)
plt.legend(loc='lower left')
plt.xlim((-0.02,1.02))
plt.ylim((-0.02,1.02))
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('(8) storage tank')
plt.savefig('./PR/storage tank_pr.png')
