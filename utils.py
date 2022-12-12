import numpy as np
from PIL import Image
import pandas as pd


import os
import glob


def get_image_paths(data_path, categories, num_train_per_cat, num_test_per_cat, test):
    '''
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default both lists will be 1500x1, where each
    entry is a char array (or string).
    '''

    num_categories = len(categories) # number of scene categories.
    print("Number of Categories: "+ str(num_categories))

    # This paths for each training and test image. By default it will have 800
    # entries (8 categories * 100 training and test examples each)
    train_image_paths = [None] * (num_categories * num_train_per_cat)
    test_image_paths  = [None] * (num_categories * num_test_per_cat)

    # The name of the category for each training and test image. With the
    # default setup, these arrays will actually be the same, but they are built
    # independently for clarity and ease of modification.
    train_labels = [None] * (num_categories * num_train_per_cat)
    test_labels  = [None] * (num_categories * num_test_per_cat)

    for i,cat in enumerate(categories):
        images = glob.glob(os.path.join(data_path, 'train', cat, '*.jpg'))
        
        for j in range(num_train_per_cat):
            train_image_paths[i * num_train_per_cat + j] = images[j]
            train_labels[i * num_train_per_cat + j] = cat
        
        images = glob.glob(os.path.join(data_path, test, cat, '*.jpg'))
        for j in range(num_test_per_cat):
            test_image_paths[i * num_test_per_cat + j] = images[j]
            test_labels[i * num_test_per_cat + j] = cat
        

        

    return (train_image_paths, test_image_paths, train_labels, test_labels)

def get_test_image_paths(data_path, categories, num_test_per_cat, perturbation, intensity):


    num_categories = len(categories) # number of scene categories.
    print("Number of Categories: "+ str(num_categories))

    # This paths for each training and test image. By default it will have 800
    # entries (8 categories * 100 training and test examples each)
    
    test_image_paths  = [None] * (num_categories * num_test_per_cat)

    # The name of the category for each training and test image. With the
    # default setup, these arrays will actually be the same, but they are built
    # independently for clarity and ease of modification.

    test_labels  = [None] * (num_categories * num_test_per_cat)

    for i,cat in enumerate(categories):
        
        images = glob.glob(os.path.join(data_path, 'test', perturbation, str(intensity), cat, '*.jpg'))
        for j in range(num_test_per_cat):
            test_image_paths[i * num_test_per_cat + j] = images[j]
            test_labels[i * num_test_per_cat + j] = cat
        
    return (test_image_paths, test_labels)





def read_img(path, mono=False):
    if mono:
        return read_img_mono(path)
    img = Image.open(path)
    return np.asarray(img)


def read_img_mono(path):
    # The L flag converts it to 1 channel.
    img = Image.open(path).convert(mode="L")
    return np.asarray(img)


def resize_img(ndarray, size):
    # Parameter "size" is a 2-tuple (width, height).
    img = Image.fromarray(ndarray.clip(0, 255).astype(np.uint8))
    return np.asarray(img.resize(size))


def rgb_to_gray(ndarray):
    gray_img = Image.fromarray(ndarray).convert(mode="L")
    return np.asarray(gray_img)


def display_img(ndarray):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).show()


def save_img(ndarray, path):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).save(path)
