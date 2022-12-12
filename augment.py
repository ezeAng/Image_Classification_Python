from logging import exception
from tkinter.tix import IMAGE
import torch 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import glob
import numpy as np
from PIL import Image 
import pandas as pd
from matplotlib import pyplot as plt


'''
Augments current images in the train folders by a random transformation. Effectively doubles the size of our dataset
'''


IMAGE_CATEGORIES = [
    'black widow', 'captain america', 'doctor strange', 'hulk', 'ironman', 'loki', 'spider-man', 'thanos'
]
DATA_PATH = 'data'


num_categories = 8
num_train_per_cat = 2
image_container = []
train_image_paths = [None] * (num_categories * num_train_per_cat)
train_labels = [None] * (num_categories * num_train_per_cat)


#Load Data
my_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ], p=0.5),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0]) CHANGE VALUES
    # #find mean and std first then inpt here
])


category_folder_path = os.path.join(DATA_PATH, 'train')
cat_folder_aug = datasets.ImageFolder(root = category_folder_path, transform = my_transform)
#Save to folder
done = ['black widow', 'captain america', 'doctor strange', 'hulk', 'ironman', 'loki', 'spider-man', 'thanos']
img_num = 0
for img, label in cat_folder_aug:
    cat = IMAGE_CATEGORIES[label]
    
    if cat in done:
        continue
    else:
        print('saving image: ', cat)
        save_image(img, category_folder_path + '/' + cat + '/aug_' + str(img_num) + '.jpg')
        img_num += 1
        

print('Data augmentation complete')
