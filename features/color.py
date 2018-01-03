#!/usr/bin/env python

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def get_RGBS(img, PLOT = False):

    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    chans = cv2.split(image)
    colors = ("r", "g", "b")
    features = []
    featuresSobel = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create histogram for the current channel and concatenate the resulting histograms for each/ hist. channel
        hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
        hist = hist/hist.sum()

        # grad_x = np.abs(cv2.Sobel(chan, cv2.CV_16S, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT))
        # grad_y = np.abs(cv2.Sobel(chan, cv2.CV_16S, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT))

        # abs_grad_x = cv2.convertScaleAbs(grad_x)
        # abs_grad_y = cv2.convertScaleAbs(grad_y)
        # dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # histSobel = cv2.calcHist([dst], [0], None, [8], [0, 256])
        # histSobel = histSobel/histSobel.sum()

        if PLOT:
            plt.subplot(2,1,1)
            plt.plot(range(0,256,32),hist,color)
            # plt.plot(range(0,256,32),histSobel,color+"--")
        
        features.extend(hist[:,0].tolist())
        # featuresSobel.extend(histSobel[:,0].tolist())

    features.extend(featuresSobel)
    Grayscale = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    histG = cv2.calcHist([chan], [0], None, [8], [0, 256])
    histG = histG / histG.sum()
    features.extend(histG[:,0].tolist())

    grad_x = np.abs(cv2.Sobel(chan, cv2.CV_16S, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT))
    grad_y = np.abs(cv2.Sobel(chan, cv2.CV_16S, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT))

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    histSobel = cv2.calcHist([dst], [0], None, [8], [0, 256])
    histSobel = histSobel/histSobel.sum()
    features.extend(histSobel[:,0].tolist())


    # calculate HSV histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    chans = cv2.splot(hsv) # take the S channel
    S = chans[1]
    hist2 = cv2.calcHist([S], [0], None, [8], [0, 256])
    hist2 = hist2/hist2.sum()
    features.extend(hist2[:,0].tolist())

    if PLOT:
        plt.subplot(2,1,2); plt.plot(range(0,256,32), hist2)
        plt.show()

    Fnames = ["Color-R"+str(i) for i in range(8)]
    Fnames.extend(["Color-G"+str(i) for i in range(8)])
    Fnames.extend(["Color-B"+str(i) for i in range(8)])
    Fnames.extend(["Color-SobelR"+str(i) for i in range(8)])
    Fnames.extend(["Color-SobelG"+str(i) for i in range(8)])
    Fnames.extend(["Color-SobelB"+str(i) for i in range(8)])
    Fnames.extend(["Color-Gray"+str(i) for i in range(8)])
    Fnames.extend(["Color-GraySobel"+str(i) for i in range(8)])
    Fnames.extend(["Color-Satur"+str(i) for i in range(8)])

    return features, Fnames