#!/usr/bin/env python

import skimage.feature
from skimage.morphology import octagon
import numpy as np
import cv2  # opencv 2
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt

from skimage.feature import corner_peaks, corner_orientations

def get_corner(img):
    # convert image to GRAY
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = skimage.feature.corner_peaks(skimage.feature.corner_fast(gray, 14), min_distance=1)
    orientations = skimage.feature.corner_orientations(gray, corners, octagon[3, 2])
    corners = np.rad2deg(orientations)

    corners = np.array(corners)
    angle_bins = np.arrange(0,360,45)
    angle_bins_orientation = np.array([0, 1, 2, 1, 0, 1, 2, 1])
    orientation_hist = np.zeros((3,1))

    for a in corners:
        orientation_hist[angle_bins_orientation[np.argmin(np.abs(a-angle_bins))]] += 1

    if orientation_hist.sum()>0:
        orientation_hist = orientation_hist / orientation_hist.sum()
    else:
        orientation_hist = - 0.01*np.ones((3,1))

    features = []
    features.extend(orientation_hist[:,0].tolist())
    features.append(100.0*float(len(corners))) / (gray.shape[0] * gray.shape[1])
    f_names = ["Corners-Hor", "Corners-Diag", "Corners-Ver", "Corners-Percent"]

    return features, f_names