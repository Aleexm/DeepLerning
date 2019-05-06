#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
from random import randint
import math

def bgr2rgb(images):
    # Convert array of images from bgr to rgb.
    rgb_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_images.append(image)
    return rgb_images


def load_images_from_folder(folder):
    # Load all images from a folder.
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# images = load_images_from_folder("augmented_image/Train")

def augment_motion_blur(img, size):
    # Add motion blur to the image in the form of horizontal pixel averaging
    #
    # img: an array of mulitple images.
    # size: size of the convolution kernel, larger size means more motion blur.
    
    output = []
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    
    for image in img:
        # applying the kernel to the input image
        blurred_img = cv2.filter2D(image, -1, kernel_motion_blur)
        blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        output.append(blurred_img)
    return output

# blurred_img = augment_motion_blur(images, 5)
# plt.imshow(blurred_img[20])
# plt.show()
# plt.imshow(images[20])
# plt.show()

def augment_occlusion(img, width_perc):
    # Add occlusion to an image in the form of a vertical bar.
    #
    # img: an array containing multiple images.
    # width_perc: percentage of image occluded by bar.
    
    output = []
    for image in images:
        imgW = image.shape[1]
        imgH = image.shape[0]
        width = imgW*width_perc/100
        pos = randint(math.ceil(width/2), image.shape[1]-math.ceil(width/2))
        occluded_img = np.copy(image)
        if width >= imgW:
            print ("Occlusion is larger than the image.")
        for x in range(pos-int(width/2), pos + math.ceil(width/2)):
            for y in range(imgH):
                occluded_img[y,x] = 0
        image = cv2.cvtColor(occluded_img, cv2.COLOR_BGR2RGB)
        output.append(image)
    return output

# image_occ = augment_occlusion(images, 10)
# plt.imshow(image_occ[1])
# plt.show()
# plt.imshow(images[1])

def augment_brightness(image,start_brightness,stop_brightness):
    # Change brightness of an image by changing the lightness in HLS format
    #
    # image: an array of multiple images.
    # start_brightness: lower bound of brightness fraction.
    # stop_brightness: upper bound of brightness fraction.
    
    output = []
    for image in images:
        #changing the color space from rgb to hsv
        image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        image1 = np.array(image1, dtype = np.float64)
        random_bright = np.random.uniform(start_brightness,stop_brightness)
        image1[...,1] = image1[...,1]*random_bright
        image1[...,1][image1[...,1]>255]  = 255
        image1 = np.array(image1, dtype = np.uint8)
        image1 = cv2.cvtColor(image1,cv2.COLOR_HLS2RGB)
        output.append(image1)
    return output

# image_ill = augment_brightness(images, 0.1, 5)
# plt.imshow(image_ill[20])
# plt.show()
# plt.imshow(images[20])



images = load_images_from_folder(data)