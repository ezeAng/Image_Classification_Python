from cgi import test
import joblib
import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.signal import convolve2d, correlate2d
from skimage.util import random_noise
import torchvision
import random
import cv2
import torchvision.transforms as transforms
from svm import DATA_PATH
from utils import display_img, read_img
import PIL.Image as Image
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import read_img_mono, display_img, save_img, read_img, get_test_image_paths
from torchmetrics import F1Score
import numpy as np
from PIL import Image
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from svm import bag_of_words


import os
import glob




'''
Gaussian pixel noise
To each pixel, add a Gaussian distributed random number with 10 increasing standard deviations from
{0, 2, 4, 6, 8, 10, 12, 14, 16, 18 }. Make sure that the pixel values are integers in the range 0..255 (e.g.
replace negative numbers by 0, values > 255 by 255).
'''
def gaussian_pixel_noise(image_paths, image_labels, intensity, mean):
    standardDeviations = [0,2,4,6,8,10,12,14,16,18]
    std = standardDeviations[intensity]

    imagesAsArraysWithPixelNoise = []
    
    for image in image_paths:
        
        imageAsArray = read_img(image)
        height,width,a = imageAsArray.shape
        new_image = np.zeros((height,width,a))
        for row in range(height):
            for column in range(width):
                pixel = imageAsArray[row][column]                
                for channel in range(3):

                    randomNumber = np.random.normal(mean, std)
                    
                    value = pixel[channel]
                    new_val = value + randomNumber
                    
                    if new_val > 255: new_val = 255
                    if new_val < 0: new_val = 0

                    new_image[row][column][channel] = new_val
        imagesAsArraysWithPixelNoise.append(new_image)

    
    return imagesAsArraysWithPixelNoise, image_labels


'''
Gaussian blurring
Create test images by blurring the original image by 3x3 mask:
Repeatedly convolve the image with the mask 0 times, 1 time, 2 times, ... 9 times. This approximates
Gaussian blurring with increasingly larger standard deviations.
'''
def gaussian_blurring(image_paths, image_labels, intensity):

    convolve_count = intensity

    imagesAsArraysWithBlurring = [] 

    BLUR = (1/16) * np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]])

    for image in image_paths:
        
        img = read_img(image)
        h, w, c = img.shape
        new_image_r = np.zeros((h,w,1))
        new_image_g = np.zeros((h,w,1))
        new_image_b = np.zeros((h,w,1))

        for row in range(h):
            for col in range(w):
                for channel in range(c):
                    if channel == 0:
                        #red
                        new_image_r[row][col] = img[row][col][0]
                    elif channel == 1:
                        #g
                        new_image_g[row][col] = img[row][col][1]
                    else:
                        #b
                        new_image_b[row][col] = img[row][col][2]
          

        image_blurry_r = new_image_r
        image_blurry_g = new_image_g
        image_blurry_b = new_image_b

        for i in range(convolve_count):
            image_blurry_r = cv2.filter2D(image_blurry_r, -1, BLUR)
            image_blurry_g = cv2.filter2D(image_blurry_g, -1, BLUR)
            image_blurry_b = cv2.filter2D(image_blurry_b, -1, BLUR)
        
        final_image = np.zeros((h,w,3))

        #Combine the 3 channels into 1 image
        for row in range(h):
            for col in range(w):
                for i in range(3):
                    if i == 0:
                        final_image[row][col][i] = image_blurry_r[row][col]
                    elif i == 1:
                        final_image[row][col][i] = image_blurry_g[row][col]
                    else:
                        final_image[row][col][i] = image_blurry_b[row][col]
        
        imagesAsArraysWithBlurring.append(final_image)

    return imagesAsArraysWithBlurring, image_labels


'''

Salt and Pepper Noise
To each test image, add salt and pepper noise of increasing strength. Essentially replace the amount in
skimage.util.random_noise(...) with {0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18}

Returns lists of salted/peppered images (255x255x3) with their corresponding labels
'''
def salt_and_pepper(image_paths, image_labels, intensity):
    amt = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    amount_ = amt[intensity]
    seasoned_images = []
    
    for each_image in image_paths:   
        img = read_img(each_image)
        img_sp_noise = random_noise(img, mode='s&p', seed=None, clip=True, amount = amount_)
        img_sp_noise = (255*img_sp_noise).astype(np.uint8)
        seasoned_images.append(img_sp_noise)

    return (seasoned_images, image_labels)


'''
Image Contrast Increase
Create test images by multiplying each pixel by { 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15,
1.20, 1.25 }. Make sure that the pixel values are integers in the range 0..255 (e.g. replace > 255 values
by 255).

'''
def image_contrast_increase(image_paths, image_labels, intensity):
    contrasts = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
    to_contrast = contrasts[intensity]

    contrasted_images = []


    for each_image in image_paths:
        img = read_img(each_image)

        
        h, w, channels = img.shape
        new_image = np.zeros((h,w,channels))
        for row in range(h):
            for col in range(w):
                for channel in range(channels):
                    pixel_channel_intensity = img[row][col][channel]
                    
                    new_val = pixel_channel_intensity * to_contrast
                    if new_val > 255 : new_val = 255
                    if new_val < 0 : new_val = 0
                    new_image[row][col][channel] = new_val
    
        contrasted_images.append(new_image)        
    
    # display_img(read_img(image_paths[0]))
    # display_img(contrasted_images[0])
    
    return contrasted_images, image_labels


'''
Image Contrast Decrease
Create test images by multiplying each pixel by { 1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10 }.
'''
def image_contrast_decrease(image_paths, image_labels, intensity):
    contrasts = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    to_contrast = contrasts[intensity]

    contrasted_images = []


    for each_image in image_paths:
        img = read_img(each_image)

        
        h, w, channels = img.shape
        new_image = np.zeros((h,w,channels))
        for row in range(h):
            for col in range(w):
                for channel in range(channels):
                    pixel_channel_intensity = img[row][col][channel]
                    
                    new_val = pixel_channel_intensity * to_contrast
                    if new_val > 255 : new_val = 255
                    if new_val < 0 : new_val = 0
                    new_image[row][col][channel] = new_val
    
        contrasted_images.append(new_image)        
    
    return contrasted_images, image_labels




'''
Occlusion of the Image Increase
In each test image, replace a randomly placed square region of the image by black pixels with square
edge length of { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45 }.
'''

def occlusion_image_increase(image_paths, image_labels, intensity):
    square_edge_lengths = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    sq_edge_len = square_edge_lengths[intensity]

    occluded_images = []

    for each_image in image_paths:
        img = read_img(each_image)


        h, w, _ = img.shape
        
        max_h = h - 1 - sq_edge_len
        max_w = w - 1 - sq_edge_len

        rand_start_x = random.randint(0,max_h)
        rand_start_y = random.randint(0,max_w)

        
        end_x = rand_start_x + sq_edge_len
        end_y = rand_start_y - sq_edge_len
        out_image = cv2.rectangle(img, (rand_start_x, rand_start_y), (end_x, end_y), (0, 0, 0), cv2.FILLED)

        occluded_images.append(out_image)
    
    return occluded_images, image_labels
                


'''
Image Brightness Decrease
Create test images by subtracting from each pixel: { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45 }. Make sure
that the pixel values are integers in the range 0..255 (e.g. replace < 0 values by 0).
'''
def image_brightness_decrease(image_paths, image_labels, intensity):
    subtractions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    to_subtract = subtractions[intensity]

    darkened_images = []


    for each_image in image_paths:
        img = read_img(each_image)

        
        h, w, channels = img.shape
        new_image = np.zeros((h,w,channels))
        for row in range(h):
            for col in range(w):
                for channel in range(channels):
                    pixel_channel_intensity = img[row][col][channel]
                    new_val = pixel_channel_intensity - to_subtract
                    if new_val > 255 : new_val = 255
                    if new_val < 0 : new_val = 0
                    new_image[row][col][channel] = new_val
    
        darkened_images.append(new_image)          
    
    return darkened_images, image_labels


'''
Image Brightness Increase
Create test images by subtracting from each pixel: { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45 }. Make sure
that the pixel values are integers in the range 0..255 (e.g. replace < 0 values by 0).
'''
def image_brightness_increase(image_paths, image_labels, intensity):
    additions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    to_add = additions[intensity]

    brightened_images = []


    for each_image in image_paths:
        img = read_img(each_image)

        
        h, w, channels = img.shape
        new_image = np.zeros((h,w,channels))
        for row in range(h):
            for col in range(w):
                for channel in range(channels):
                    pixel_channel_intensity = img[row][col][channel]
                    new_val = pixel_channel_intensity + to_add
                    if new_val > 255 : new_val = 255
                    if new_val < 0 : new_val = 0
                    new_image[row][col][channel] = new_val
                    
        print('Done with an image')
        brightened_images.append(new_image)        
    
    return brightened_images, image_labels


#List of perturbations and respective functions
perturbations = [
    # ('gaussian_pixel_noise', gaussian_pixel_noise), #DONE
    # ('gaussian_blurring', gaussian_blurring), #DONE
    # ('image_contrast_increase', image_contrast_increase), #DONE
    # ('image_contrast_decrease', image_contrast_decrease), #DONE
    # ('image_brightness_increase', image_brightness_increase), #DONE
    # ('image_brightness_decrease', image_brightness_decrease), #DONE
    # ('occlusion_image_increase', occlusion_image_increase), #DONE
    # ('salt_and_pepper', salt_and_pepper) #DONE
    
]


'''
Loops through each pertubation, given the data path containing the test-images
returns a dictionary mapping perturbation -> list of list of perturbed images with varying intensity
Should just do it once before throwing into testing.
'''
def get_perturbed_images(pertubations, test_image_paths, test_labels):
    print('Generating Perturbed Images...')

    #ocluson: [([imgs..], 0, labels), ([imgs..], 1, labels), ....]
    perturbed = {

    }
    for each in pertubations:
        
        perturbed[each[0]] = []
        for intensity in range(10):
            print('Generating ' + each[0] + 'Intensity ' + str(intensity))
            if each[0] == 'gaussian_pixel_noise':
                perturbed_imgs, perturbed_labels = each[1](test_image_paths, test_labels, intensity, 127)
            else:
                perturbed_imgs, perturbed_labels = each[1](test_image_paths, test_labels, intensity)
            perturbed[each[0]].append((perturbed_imgs, intensity, perturbed_labels))
    
    print('Done Pertubations')
    
    
    return perturbed


'''
Saves perturbed images into destination folders to be used for testing
'''
def save_perturbed_images(perturbed_dict): 
    for perturbation, perturbed_images_over_intensities in perturbed_dict.items():
        for perturbed_images_tuple in (perturbed_images_over_intensities):
            
            perturbed_imgs = perturbed_images_tuple[0]
            perturbed_intensity = perturbed_images_tuple[1]
            perturbed_labels = perturbed_images_tuple[2]
            

            for i,img in enumerate(perturbed_imgs):
                cat = perturbed_labels[i]

                file_path = DATA_PATH + '/test/' + perturbation +  '/' + str(perturbed_intensity) + '/' + cat + '/pert' + str(i) + '.jpg'

                # display_img(img)
                save_img(img, file_path)


### ------ START TESTING HERE -------



# Standard field
BATCH_SIZE = 32
DATA_PATH = 'data'
IMAGE_CATEGORIES = [
    'black widow', 'captain america', 'doctor strange', 'hulk', 'ironman', 'loki', 'spider-man', 'thanos'
]
#Gets image paths from data/test/default/0
test_image_paths, test_labels = get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES, 10, 'default', '0')

# ### ------- GENERATE PERTURBED IMAGES -------
# perturbed_dict = get_perturbed_images(perturbations, test_image_paths, test_labels)

# ### ------- Saves perturbed images of the built dictionary -------
# save_perturbed_images(perturbed_dict)




