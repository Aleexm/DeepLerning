#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
from random import randint
import math

def bgr2rgb(images):
    rgb_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_images.append(image)
    return rgb_images

def load_filenames_from_folder(folder):
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            filenames.append(filename)
    return filenames
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder(r"C:\Users\Stefan\Documents\Msc ME-VE\S2\Deep Learning\Final Project\gtsrb-german-traffic-sign/Test")
filenames = load_filenames_from_folder(r"C:\Users\Stefan\Documents\Msc ME-VE\S2\Deep Learning\Final Project\gtsrb-german-traffic-sign/Test")



plt.imshow(images[1])

plt.show()

images_rgb = bgr2rgb(images)

plt.imshow(images_rgb[1])

print(filenames[1])


# In[ ]:


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
        #blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        output.append(blurred_img)
    return output

# Create and save augmented images
def create_blurred_test_set(size):
    blurred_img = augment_motion_blur(images, size)

    path = r"C:\Users\Stefan\Documents\Msc ME-VE\S2\Deep Learning\Final Project\gtsrb-german-traffic-sign/Test\Blurred_" + str(size)

    for it in range(len(blurred_img)):
        cv2.imwrite(os.path.join(path, filenames[it]), blurred_img[it])
        
create_blurred_test_set(5)
create_blurred_test_set(10)
create_blurred_test_set(15)


# In[9]:


def augment_occlusion(img, width_perc):
    # Add occlusion to an image in the form of a vertical bar.
    #
    # img: an array containing multiple images.
    # width_perc: percentage of image occluded by bar.
    
    output = []
    for image in images:
        imgH, imgW, _ = image.shape
        width = imgW*width_perc/100
        pos = randint(width//2, image.shape[1] - width//2)
        occluded_img = np.copy(image)
        if width >= imgW:
            print ("Occlusion is larger than the image.")
        for x in range(pos- int(width // 2), pos + int(width//2)):
            for y in range(imgH):
                occluded_img[y,x] = 0
#         image = cv2.cvtColor(occluded_img, cv2.COLOR_BGR2RGB)
        output.append(occluded_img)
    return output

# EXAMPLE:
# image_occ = augment_occlusion(images, 10)
# plt.imshow(image_occ[1])
# plt.show()
# plt.imshow(images[1])
# plt.show()

# Create and save augmented images
def create_occluded_test_set(size):
    occl_img = augment_occlusion(images, size)

    path = r"C:\Users\Stefan\Documents\Msc ME-VE\S2\Deep Learning\Final Project\gtsrb-german-traffic-sign/Test\Occl_" + str(size)

    for it in range(len(occl_img)):
        cv2.imwrite(os.path.join(path, filenames[it]), occl_img[it])
        

create_occluded_test_set(5)
print('done')
create_occluded_test_set(10)
print('done')
create_occluded_test_set(15)
print('done')
create_occluded_test_set(20)
print('done')
create_occluded_test_set(25)
print('done')


# In[12]:


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
        #image1 = cv2.cvtColor(image1,cv2.COLOR_HLS2RGB)
        image1 = cv2.cvtColor(image1,cv2.COLOR_HLS2BGR) # For saving test sets.
        output.append(image1)
    return output

# EXAMPLE:
# image_ill = augment_brightness(images, 0.1, 5)
# plt.imshow(image_ill[20])
# plt.show()
# plt.imshow(images[20])




images = load_images_from_folder(data)

# Create and save augmented images
def create_brightness_test_set(start, end, number):
    bright_img = augment_brightness(images, start, end)

    path = r"C:\Users\Stefan\Documents\Msc ME-VE\S2\Deep Learning\Final Project\gtsrb-german-traffic-sign/Test\Bright_" + str(number)

    for it in range(len(bright_img)):
        cv2.imwrite(os.path.join(path, filenames[it]), bright_img[it])

create_brightness_test_set(.1,.4,1)
print('done')
create_brightness_test_set(.4,.7,2)
print('done')
create_brightness_test_set(.7,1,3)
print('done')
create_brightness_test_set(1,1.3,4)
print('done')
create_brightness_test_set(1.3,1.6,5)
print('done')
create_brightness_test_set(1.6,1.9,6)
print('done')
create_brightness_test_set(1.9,2.2,7)
print('done')
create_brightness_test_set(2.2,2.5,8)
print('done')
create_brightness_test_set(2.5,2.8,9)
print('done')
create_brightness_test_set(2.8,3.1,10)
print('done creating test set.')

