# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:20:53 2015

"""

import cv2
import numpy as np


im2 = cv2.imread('input/trees.jpg')

### darkchannel
def darkchannel(image_array):
    """
    Calculates the darkchannel, a 2D array of same size as picture.
    
    The value of a darkchannel pixel is the lowest intesity pixel in the patch.
    """
    height, width, rgb = image_array.shape
    patchSize = 15
    padSize = 7
    JDark = np.zeros((height,width))
    imJ = np.pad(image_array, ((padSize, padSize),(padSize, padSize), (0,0)), mode='constant', constant_values=(255) )
    for r in xrange(height):
        for c in xrange(width):
            patch = imJ[r:(r+patchSize-1), c:(c+patchSize-1),:]
            JDark[r,c] = patch.min()
    return JDark


### Atmospheric light
def atmospheric_light(image_array, JDark):
    height, width, rgb = image_array.shape
    numpx = width*height / 1000
    print "Dom pixel count", numpx
    if numpx == 0:
        numpx = 1
    JDarkVec = np.reshape(JDark, (width*height))
    ImVec = np.reshape(image_array, (width*height, 3))
    indices = np.argsort(JDarkVec)
    brightest_indices = indices[width*height-numpx:]
    atmSum = np.zeros(3)
    pixel_w = []
    pixel_h = []
    for i in xrange(numpx):
        atmSum = atmSum + ImVec[brightest_indices[i]]
        pixel_w.append(i%width)
        pixel_h.append(int(i/width)+1)
    A = atmSum/numpx
    return A


### Transmission Estimate
def transmission_estimate(image_array, A):
    omega = 0.95
    im3 = np.zeros(image_array.shape)
    for x in [0,1,2]:
        im3[:,:,x] = image_array[:,:,x]/float(A[x])
    transmission = 1 - omega*darkchannel(im3)
    return transmission


def get_radiance(A, image_array, tMat):
    t0 = 0.1
    J = np.zeros(image_array.shape)
    for i in [0,1,2]:
        J[:,:,i] = A[i] + (image_array[:,:,i] - A[i])/np.maximum(tMat, t0)
    return J


print "\nMake dark channel"
im2_jdark = darkchannel(im2)

print "Dom", im2_jdark[100,100]

print "\nDetemine atmospheric light"
im2_A = atmospheric_light(im2, im2_jdark)

print "\nCalculate transmission"
im2_transmission = transmission_estimate(im2, im2_A)

print "\nRecover radiance"
im2_out = get_radiance(im2_A, im2, im2_transmission)
cv2.imwrite('dehazed_output.jpg', im2_out)



