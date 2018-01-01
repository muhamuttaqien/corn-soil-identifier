#!/usr/bin/env python
"""
written by: Muhammad Angga Muttaqien
"""

import argparse
import os

import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from scipy.misc import imsave

from preprocessing import convert_to_gray, threshold, remove_noise, contrast, convolve, equalize_histogram, open_morph, close_morph
from histogram import analyze_histogram, plot_histogram, analyze_color_histogram, plot_color_histogram
from features import get_mean_area


def load_image(path, resize):
    """
    load soil image and resize proportionally 
    """
    img = cv2.imread(path)
    height, width, _ = img.shape
    img = cv2.resize(img, (int(round(width/resize)), int(round(height/resize))))

    return img


def segment_object(img):
    """
    segment object to get region of interest using connected components and finding countures techniques
    """
    # find connected components
    labeled, nr_objects = ndimage.label(img)

    #count the size
    component_size = np.bincount(labeled.flat)
    big_component = np.where(component_size > 200)

    big = np.zeros(shape=labeled.shape)

    #delete small components
    for i in range(1,len(big_component[0])):
        index = np.where(labeled == big_component[0][i])
        big[index] = i
        
    big_count = len(big_component[0])

    result = np.where(big > 0, 255,0)

    result = np.array(result, dtype=np.uint8)

    return result

def get_processed_image(img, threshVal, kernelSize):
    """
    preprocess image before segmented
    """
    img = contrast(img)
    img = remove_noise(img)
    img = convert_to_gray(img)
    img = equalize_histogram(img)
    ret, img = threshold(img, threshVal)
    img = segment_object(img)

    result = close_morph(img, kernelSize)

    imsave("./roi/img-thresh" + str(threshVal) + "-close" + str(kernelSize) + ".jpg", result)

    return result


def main():
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./soilsection/DSCN3342.JPG')
    parser.add_argument('--resize', default=6, type=int)
    parser.add_argument('--threshVal', default=127, type=int)
    parser.add_argument('--kernelSize', default=10, type=int)
    args = parser.parse_args()

    path = args.path
    resize = args.resize
    threshVal = args.threshVal
    kernelSize = args.kernelSize

    print(args)

    img = load_image(path, resize)
    result = get_processed_image(img, threshVal, kernelSize)

    cv2.imshow('Original Image', img)
    cv2.imshow('Result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()