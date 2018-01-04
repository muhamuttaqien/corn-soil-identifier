#!/usr/bin/env python

import cv2
import math
import numpy as np
from scipy import ndimage
from scipy.stats import skew
from matplotlib import pyplot as plt

from scipy.misc import imsave


# COLOR AVERAGE

img = cv2.imread('./fruit.jpeg')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

skew = skew(img_rgb)

print(skew)