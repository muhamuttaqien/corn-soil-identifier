#!/usr/bin/env python

import cv2
import math
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from scipy.misc import imsave


# COLOR AVERAGE

img = cv2.imread('./fruit.jpeg')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

chans = cv2.split(img_rgb)
colors = ("hue", "saturation", "value")
features = []


# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create histogram for each channel
    hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
    hist = hist/hist.sum()

    features.append({ color: hist })


print(features)