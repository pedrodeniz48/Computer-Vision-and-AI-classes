#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def filter(img, kernel):
    r, c = img.shape
    img_float = img.astype('float')
    img_filtered = np.zeros(shape=(r, c), dtype='float')

    ker_h, ker_w = kernel.shape

    kernel = kernel/(ker_h*ker_w)

    for row in range(ker_h, r-(ker_h-1)):
        for col in range(ker_w, c-(ker_w-1)):
            filt_val = np.sum(kernel * img_float[row:row+ker_h, col:col+ker_w])
            img_filtered[row, col] = filt_val

    img_filtered = img_filtered.astype('uint8')
    
    return img_filtered

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel = np.array([[1, 0, 1],
                       [2, 0, 2],
                       [1, 0, 1]])
    
    img_filtered = filter(img_gray, kernel)

    #Opencv filter
    kernel = kernel/(kernel.shape[0]*kernel.shape[1])
    img_opencv = cv.filter2D(img_gray, -1, kernel)

    #Graph
    fig1, ax1 = plt.subplots(1, 2)
    ax1[0].imshow(img_filtered, cmap='gray', vmin=0, vmax=255)
    ax1[0].set_title('Filter function')
    ax1[1].imshow(img_opencv, cmap='gray', vmin=0, vmax=255)
    ax1[0].set_title('Opencv filter function')


    # Derivative kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    img_der_x = cv.filter2D(img_gray, -1, kernel_x)
    img_der_y = cv.filter2D(img_gray, -1, kernel_y)

    # Laplacian
    laplacian = np.abs(img_der_x) + np.abs(img_der_y)

    # Display images
    fig2, ax2 = plt.subplots(1, 4, figsize=(12, 4))
    ax2[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    ax2[0].set_title('Original Image')
    ax2[1].imshow(img_der_x, cmap='gray', vmin=0, vmax=255)
    ax2[1].set_title('Derivative X')
    ax2[2].imshow(img_der_y, cmap='gray', vmin=0, vmax=255)
    ax2[2].set_title('Derivative Y')
    ax2[3].imshow(laplacian, cmap='gray', vmin=0, vmax=255)
    ax2[3].set_title('Laplacian')

    fig1.savefig('images/Filers_Comparation')
    fig2.savefig('images/Derivatives_Laplacian')

    plt.show()