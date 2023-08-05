# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:10:17 2020

@author: alfah
"""
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ct_scan_nifti = nib.load('049.nii')
ct_scan = ct_scan_nifti.get_data()
def window_image(img, window_center, window_width):

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

image = window_image(ct_scan[:,:,20], 40, 380)
mini = np.min(image)
maxi = np.max(image)
img = (image - mini)*(255/(maxi-mini))
imgsoft = np.uint8(img)
plt.imshow(imgsoft,cmap='gray')

