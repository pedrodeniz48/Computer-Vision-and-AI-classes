#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')

    r, c, ch = img.shape
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Image equalized
    img_eq = cv.equalizeHist(img_gray)

    #Histograms calculation
    hist_gray = cv.calcHist(img_gray, [ch], None, [256], [0, 256])

    hist_eq = cv.calcHist(img_eq, [ch], None, [256], [0, 256])
    
    #X-axis values
    bins = [idx for idx in range(256)]

    #Graph
    fig, ax = plt.subplots(2, 2)

    #Gray Image
    ax[0, 0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].bar(list(bins), list(hist_gray[:, 0]))
    
    #Image equalized
    ax[1, 0].imshow(img_eq, cmap='gray', vmin=0, vmax=255)
    ax[1, 1].bar(list(bins), list(hist_eq[:, 0]))

    fig.savefig('images/Cat_Hists_Graph')

    plt.show()