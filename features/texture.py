#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# features extracted from textures are
# texture: mean, standard deviation, energy, entropy, skewness, variance, homogenity, kurtosis, smoothness
# geometric: area, perimeter, eccentricity, orientation, convexity, equivalent diameter, compactness

# color: mean, standard deviation, skewness, color histogram, color average
# corner
# shape: edge detection
# texture: mean, standard deviation, GLCM (scikit-image), local binary patterns (scikit-image)