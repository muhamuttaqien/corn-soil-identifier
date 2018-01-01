import cv2
import numpy as np


def convert_to_gray(img):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def threshold(img):

    ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return ret, img

def remove_noise(img):

    img = cv2.medianBlur(img,5)
    return img

def contrast(img):

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def convolve(img):
    pass


def equalize_histogram(img):
   
    img = cv2.equalizeHist(img)
    # res = np.hstack((img,equ) #stacking images side-by-side
    return img