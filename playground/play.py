import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from scipy.misc import imsave

# CONNECTED COMPONENT

img1 = cv2.imread('./depok2016.jpg',0)
imsave('./depok-img1.jpg', img1)
img2 = cv2.imread('./depok2005.jpg',0)
imsave('./depok-img2.jpg', img2)

sub = abs(img2 - img1)

imsave('./depok-diff.jpg', sub)

thres = np.where(sub <= 80, 0, 255)

imsave('./depok-diff80.jpg', thres)

# find connected components
labeled, nr_objects = ndimage.label(thres) 

#count the size
component_size = np.bincount(labeled.flat)
big_component = np.where(component_size > 1900)

big = np.zeros(shape=labeled.shape)

#delete small components
for i in range(1,len(big_component[0])):
    index = np.where(labeled == big_component[0][i])
    big[index] = i
	   
big_count = len(big_component[0])

result = np.where(big > 0, 255,0)

result = np.array(result, dtype=np.uint8)

imsave('./depok-diff80-s1900.jpg', result)

kernel = np.ones((10,10),np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

imsave('./depok-diff80-s1900-closing.jpg', result)