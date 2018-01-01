#!/usr/bin/env python
"""
written by: Muhammad Angga Muttaqien
"""

import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from preprocessing import convert_to_gray, threshold, remove_noise, contrast, convolve, equalize_histogram



def analyze_histogram(img):
    """
    find and analyze histogram
    """
    # convert first
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # analyze it
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    return hist


def plot_histogram(img):
    """
    plot and analyze histogram for gray image
    """
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    plt.hist(hist, facecolor='green')
    plt.title('Histogram'), plt.xlabel("Scale"), plt.ylabel("Quantity")
    plt.grid(True)

    plt.show()


def plot_color_histogram(img):
    """
    plot and analyze histogram for color image
    """
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        
        plt.hist(hist, color = col)
        plt.xlim([0,256])
    
    plt.title('Color Histogram'), plt.xlabel("Scale"), plt.ylabel("Quantity")
    plt.grid(True)

    plt.show()


def load_image(path, resize):
    """
    load soil image and resize proportionally 
    """
    img = cv2.imread(path)
    height, width, _ = img.shape
    img = cv2.resize(img, (round(width/resize), round(height/resize)))

    return img


def get_connected_components():
    pass


def get_processed_image(img):
    """
    preprocess image before segmented
    """
    img = contrast(img)
    img = remove_noise(img)
    img = convert_to_gray(img)
    result = equalize_histogram(img)
    ret, _ = threshold(img)

    return ret, result


def main():
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./soilsection/DSCN3342.JPG')
    parser.add_argument('--resize', default=6, type=int)
    args = parser.parse_args()

    path = args.path
    resize = args.resize

    img = load_image(path, resize)
    plot_color_histogram(img)
    ret, result = get_processed_image(img)

    cv2.imshow('Original Image', img)
    cv2.imshow('Result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()