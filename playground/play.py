import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from scipy.misc import imsave

# CONNECTED COMPONENT

img = cv2.imread('./rectangle.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
result, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)

imsave('./rectangle-with-contours.jpg', img)