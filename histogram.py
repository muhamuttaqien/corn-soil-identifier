#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def analyze_histogram(img):
    """
    find and analyze histogram
    """
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    return hist


def plot_histogram(img):
    """
    plot and analyze histogram for gray image
    """
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    plt.hist(hist,facecolor='green')
    plt.title('Histogram'), plt.xlabel("Scale"), plt.ylabel("Quantity")
    plt.grid(True)

    plt.show()


def analyze_color_histogram(img):
    """
    find and analyze histogram for color image
    """
    color = ('b', 'g', 'r')
    hist = []    
    for i, col in enumerate(color):
        hist.append(cv2.calcHist([img],[i],None,[256],[0,256]))     
        
    blue = hist[0]
    green = hist[1]
    red = hist[2]
    
    return blue, green, red


def plot_color_histogram(img):
    """
    plot and analyze histogram for color image
    """
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])     
        plt.plot(hist, color = col)
        plt.xlim([0,256])
    
    plt.title('Color Histogram'), plt.xlabel("Scale"), plt.ylabel("Quantity")
    plt.grid(True)

    plt.show()