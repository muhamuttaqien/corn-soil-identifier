#!/usr/bin/env python

import cv2
import math
import numpy as np
from scipy import ndimage
from scipy.stats import skew
from matplotlib import pyplot as plt

from scipy.misc import imsave

from skimage.feature import greycomatrix, greycoprops


image = np.array([[0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 2, 2, 2],
                   [2, 2, 3, 3]], dtype=np.uint8)

glcm = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

contrast = greycoprops(glcm, prop='contrast')
dissimilarity = greycoprops(glcm, prop='dissimilarity')
homogeneity = greycoprops(glcm, prop='homogeneity')
ASM = greycoprops(glcm, prop='ASM')
energy = greycoprops(glcm, prop='energy')
correlation = greycoprops(glcm, prop='correlation')

print(contrast)
print(dissimilarity)
print(homogeneity)
print(ASM)
print(energy)
print(correlation)