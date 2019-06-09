#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:09:46 2019

@author: anlimo1510
"""

import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties

import matplotlib
import seaborn
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#font = FontProperties(fname=r"/usr/share/fonts/WinFonts/simsun.ttc", size=14)  

label_list = ['飞机', '轮船', '存储罐', '棒球场','网球场','篮球场','田径场','港口','桥梁','车辆']    # 横坐标刻度显示值

num_list = [536, 204, 501, 263, 366, 111, 108, 177, 91, 433]      # 纵坐标值

x = range(len(num_list))  # 横坐标

rects1 = plt.bar(x, num_list, width = 0.618, color=seaborn.xkcd_rgb['dark sage'])

plt.ylim(0, 600)     # y轴取值范围
#plt.legend()     # 设置题注
plt.title("图。。。。")
plt.xticks([index + 0.0 for index in x], label_list)
plt.xlabel('类别')
plt.ylabel('数量')
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

#plt.show()
plt.savefig("1.png")