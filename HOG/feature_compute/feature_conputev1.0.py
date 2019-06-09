#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:20:57 2019

@author: anlimo1510
"""

# prepare data
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# read img as grey
img = cv2.imread('67.jpg',cv2.IMREAD_GRAYSCALE)
# use Gamma to normolize the color space
img = np.sqrt(img / float(np.max(img)))
plt.imshow(img)
plt.savefig('67_grey.png')

# compute gradscaly of every pixel
height, width = img.shape
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees = True)
gradient_magnitude.shape, gradient_angle.shape
#print(gradient_magnitude.shape, gradient_angle.shape)



# construct gradient direction histogram for each cell unit
# scale of cell is defaulted as 8*8
# use 8 bin of histogram to count grandient informantion of a cell
cell_size = 4
bin_size = 8
angle_unit = 360 / bin_size
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))
#print(cell_gradient_vector.shape)

def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit) % 8
            max_angle = (min_angle + 1) % bin_size
            mod= gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - mod / angle_unit))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers


for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                                            j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                                    j * cell_size:(j + 1) * cell_size]
#        print(cell_angle.max())
        
        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)


hog_image = np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()
for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag
        angle = 0
        angle_gap = angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))            
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap

plt.imshow(hog_image, cmap=plt.cm.gray)
plt.savefig('67_hog.png')
plt.show()


# count gradient information of block
hog_vector = []
for i in range(cell_gradient_vector.shape[0] - 1):
    for j in range(cell_gradient_vector.shape[1] - 1):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i][j])
        block_vector.extend(cell_gradient_vector[i][j + 1])
        block_vector.extend(cell_gradient_vector[i + 1][j])
        block_vector.extend(cell_gradient_vector[i + 1][j + 1])
        mag = lambda vector: math.sqrt(sum(1 ** 2 for i in vector))
        magnitude = mag(block_vector)
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            block_vector = normalize(block_vector, magnitude)
        hog_vector.append(block_vector)
#print(np.array(hog_vector).shape)
# (11484, 32): totally 11484 block, with 32 dimension every block














    
