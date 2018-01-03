#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def get_rgb_avg(img):

    # convert image to RGB
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    chans = cv2.split(img_rgb)
    colors = ("red", "green", "blue")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):

        color_amount = 0

        # calculate average of each channel
        for row_value in chan:
            for col_value in row_value:
                color_amount += int(col_value)

        color_avg = color_amount/chan.size

        features.append({ color: color_avg })

    return features


def get_rgb_histogram(img):

    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector    
    chans = cv2.split(image)
    colors = ("r", "g", "b")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):

        # create histogram for each channel
        hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
        hist = hist/hist.sum()

        features.append({ color: hist })

    return features


def get_hsv_avg(img):

    # convert image to RGB
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    chans = cv2.split(img_rgb)
    colors = ("hue", "saturation", "value")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):

        color_amount = 0

        # calculate average of each channel
        for row_value in chan:
            for col_value in row_value:
                color_amount += int(col_value)

        color_avg = color_amount/chan.size

        features.append({ color: color_avg })

    return features


def get_hsv_histogram(img):

    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector    
    chans = cv2.split(image)    
    colors = ("hue", "saturation", "value")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):

        # create histogram for each channel
        hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
        hist = hist/hist.sum()

        features.append({ color: hist })

    return features