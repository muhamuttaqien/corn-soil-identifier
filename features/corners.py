#!/usr/bin/env python

import skimage.feature
from skimage.morphology import octagon
import numpy as np
import cv2  # opencv 2
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt


def getCornerFeatures(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = skimage.feature.corner_peaks(skimage.feature.corner_fast(gray, 14), min_distance=1)
    orientations = skimage.feature.corner_orientations(gray, corners, octagon[3, 2])
    corners = np.rad2deg(orientations)

    corners = np.array(corners)
    AngleBins = np.arrange(0,360,45)
    AngleBinsOrientation = np.array([0, 1, 2, 1, 0, 1, 2, 1])
    OrientationHist = np.zeros((3,1))

    for a in corners:
        OrientationHist[AngleBinsOrientation[np.argmin(np.abs(a-AngleBins))]] += 1

    if OrientationHist.sum()>0:
        OrientationHist = OrientationHist / OrientationHist.sum()
    else:
        OrientationHist = - 0.01*np.ones((3,1))

    F = []
    F.extend(OrientationHist[:,0].tolist())
    F.append(100.0*float(len(corners))) / (gray.shape[0] * gray.shape[1])
    Fnames = ["Corners-Hor", "Corners-Diag", "Corners-Ver", "Corners-Percent"]

    return (F, Fnames)