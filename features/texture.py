#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

def get_mean(img):
    pass


def get_std(img):
    pass


def get_glcm(img):
    # compute the Gray Level Co-occurence Matrix representation
    glcm = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    contrast = greycoprops(glcm, prop='contrast')
    homogeneity = greycoprops(glcm, prop='homogeneity')
    energy = greycoprops(glcm, prop='energy')
    correlation = greycoprops(glcm, prop='correlation')
    
    # return contrast, homogeneity, energy and correlation feature
    return contrast, homogeneity, energy, correlation


def get_lbp(img):    
    # compute the Local Binary Pattern representation
    lbp = local_binary_pattern(img, 24, 8, method='uniform')

    hist, _ = np.histogram(lbp.ravel(),
        bins=np.arange(0, 27), 
        range=(0, 26))
    
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
	# return the histogram of Local Binary Patterns    
    return hist
