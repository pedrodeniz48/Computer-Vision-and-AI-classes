#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')

    r, c, ch = img.shape

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_float = img_gray.astype('float')
    img_filtered = np.zeros(shape=(r,c), dtype='float')

    #Using a 3x3 kernel
    kernel = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]]
    
    for row in range(1, r-1):           #m
        for col in range(1, c-1):       #n
            filt_val = 0.0
            for krow in range(3):       #K
                for kcol in range(3):   #L
                    filt_val += kernel[krow][kcol] * img_float[row + krow - 1, col + kcol - 1]
            img_filtered[row, col] = filt_val
    
    img_filtered = img_filtered.astype('uint8')

    #Graph
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(img_filtered, cmap='gray', vmin=0, vmax=255)

    plt.show()