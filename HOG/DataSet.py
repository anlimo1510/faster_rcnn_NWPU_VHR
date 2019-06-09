#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:08:38 2019

@author: anlimo1510
"""

import numpy as np

 
class DataSet(object):
 
    def __init__(self, images, labels, num_examples):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0   # 调用next_batch()函数后记住上一次位置
        self._num_examples = num_examples  # 训练样本数
 
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epochs   # 起点为上次位置
 
        if self._epochs_completed == 0 and start == 0 and shuffle:  # 起点为0，完成epochh为0 == 第一次
            index0 = np.arange(self._num_examples)  # arange表示以_num_examples为终点，产生步长为1的列表，从0开始
            # print(index0)
            np.random.shuffle(index0)  # 打乱列表顺序
            # print(index0)
            self._images = np.array(self._images)[index0]
            self._labels = np.array(self._labels)[index0]
            #print(self._images)
            #print(self._labels)
            #print("-----------------")
 
        if start + batch_size > self._num_examples:  # 剩下未训练的样本数少于一次epoch
            self._epochs_completed += 1   # 完成的epoch加一
            rest_num_examples = self._num_examples - start   # 剩下的样本数
            images_rest_part = self._images[start:self._num_examples]  # 剩下的样本
            labels_rest_part = self._labels[start:self._num_examples]  #剩下对应的标签
            if shuffle:
                index = np.arange(self._num_examples)  # 索引为样本
                np.random.shuffle(index)  # 打乱索引
                self._images = self._images[index]
                self._labels = self._labels[index]
            start = 0  # 起点为0
            self._index_in_epochs = batch_size - rest_num_examples  #
            end = self._index_in_epochs
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
 
        else:
            self._index_in_epochs += batch_size   # 中间
            end = self._index_in_epochs
            return self._images[start:end], self._labels[start:end]
 
 
