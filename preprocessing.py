#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def convert_to_gray(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img


def threshold(img, val):
    ret, img = cv2.threshold(img,val,255,cv2.THRESH_BINARY)
    return ret, img


def remove_noise(img):
    img = cv2.medianBlur(img,5)
    return img


def contrast(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img,-1,kernel)
    return img


def convolve(img,size,divider):
    kernel = np.ones((size,size),np.float32)/divider
    img = cv2.filter2D(img,-1,kernel)
    return img


def equalize_histogram(img):
    img = cv2.equalizeHist(img)
    # res = np.hstack((img,equ) #stacking images side-by-side
    return img


def erose_morph(img, size, iter):
    kernel = np.ones((size,size),np.uint8)
    img = cv2.erode(img,kernel,iterations = iter)
    return img


def dilate_morph(img, size, iter):
    kernel = np.ones((size,size),np.uint8)
    img = cv2.dilate(img,kernel,iterations = iter)
    return img


def open_morph(img, size):
    kernel = np.ones((size,size),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
    return img


def close_morph(img, size):
    kernel = np.ones((size,size),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    return img