#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:42:34 2019

@author: anlimo1510
"""

import tensorflow as tf 
#from  tensorflow.examples.tutorials.mnist  import  input_data
import numpy as np 
#import DataSet

#mnist = input_data.read_data_sets('./data/', one_hot = True)

def trans(a):
    ds = np.zeros((1,10))
    a = int(a)
    if a == 1:
        ds[:,9] = 1
        return ds
    if a == 2:
        ds[:,8] = 1
        return ds
    if a == 3:
        ds[:,7] = 1
        return ds
    if a == 4:
        ds[:,6] = 1
        return ds
    if a == 5:
        ds[:,5] = 1
        return ds
    if a == 6:
        ds[:,4] = 1
        return ds
    if a ==7:
        ds[:,3] = 1
        return ds
    if a == 8:
        ds[:,2] = 1
        return ds
    if a == 9:
        ds[:,1] = 1
        return ds
    if a == 10:
        ds[:,0] = 1
        return ds
    
    

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


num_classes = 10  # 输出大小
input_size = 648  # 输入大小
hidden_units_size = 30  # 隐藏层节点数量
batch_size = 100  # 每批样本数
training_iterations = 4000  # 训练迭代次数

X = tf.placeholder(tf.float32, shape = [None, input_size])  #输入与输出预留
Y = tf.placeholder(tf.float32, shape = [None, num_classes])

W1 = tf.Variable(tf.random_normal ([input_size, hidden_units_size], stddev = 0.1))  # 正态分布(shape,mean,stddev,dtype,seed,name)
B1 = tf.Variable(tf.constant (0.1), [hidden_units_size])
W2 = tf.Variable(tf.random_normal ([hidden_units_size, num_classes], stddev = 0.1))
B2 = tf.Variable(tf.constant (0.1), [num_classes])

hidden_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播  # matmul两个张量相乘
hidden_opt = tf.nn.relu(hidden_opt)  # 激活函数，用于计算节点输出值  # RELU
final_opt = tf.matmul(hidden_opt, W2) + B2  # 隐藏层到输出层正向传播 

# 对输出层计算交叉熵损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
# 梯度下降算法，这里使用了反向传播算法用于修改权重，减小损失
opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 计算准确率
correct_prediction =tf.equal (tf.argmax (Y, 1), tf.argmax(final_opt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

input = np.load("/home/anlimo1510/projects/HOG/feature_compute/fds2.npy")

labelfile = '/home/anlimo1510/projects/HOG/feature_compute/sample.txt'
f = open(labelfile)
output = np.zeros((2790,10))
i = 0
for line in f.readlines():
    lineArray = line.strip().split(' ')

    output[i] = trans(lineArray[1])
    i += 1
    
f.close()

ds = DataSet(input, output, 2790)

sess = tf.Session ()
sess.run (init)
for i in range (training_iterations) :
    batch = ds.next_batch (batch_size)
    batch_input = batch[0]  #X
    batch_labels = batch[1] #Y
    # 训练
    training_loss = sess.run ([opt, loss], feed_dict = {X: batch_input, Y: batch_labels})
    if i % 100 == 0 :
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: batch_input,Y: batch_labels})
        print ("step : %d, training accuracy = %g " % (i, train_accuracy))

