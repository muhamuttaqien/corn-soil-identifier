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
    original_img = cv2.imread(path)
    height, width, _ = original_img.shape
    original_img = cv2.resize(original_img, (int(round(width/resize)), int(round(height/resize))))

    return original_img


def get_connected_components(img):
    """
    get connected components of soilsection images
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


def segment_object(original_img, processed_img):
    """
    segment object to get region of interest by finding countures techniques
    """
    result, contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_img, contours, -1, (0,255,0), 3)
    
    return result, contours, hierarchy


def get_processed_image(original_img, threshVal, kernelSize):
    """
    preprocess image before segmented
    """
    imsave("./roi/img-original.jpg", original_img)
    
    processed_img = contrast(original_img)
    processed_img = remove_noise(processed_img)
    processed_img = convert_to_gray(processed_img)
    processed_img = equalize_histogram(processed_img)
    ret, processed_img = threshold(processed_img, threshVal)
    processed_img = get_connected_components(processed_img)
    processed_img = close_morph(processed_img, kernelSize)
    imsave("./roi/img-thresh" + str(threshVal) + "-close" + str(kernelSize) + ".jpg", processed_img)

    result, contours, hierarchy = segment_object(original_img, processed_img)
    imsave("./roi/img-thresh" + str(threshVal) + "-close" + str(kernelSize) + "-segmented" + ".jpg", original_img)
    
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

    original_img = load_image(path, resize)
    result = get_processed_image(original_img, threshVal, kernelSize)    

    cv2.imshow('Original Image', original_img)
    cv2.imshow('Result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()