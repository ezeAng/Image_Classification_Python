from cProfile import label
from cgi import test
import os
from pyexpat import model
import numpy as np
import PIL.Image as Image
from   matplotlib import image, markers, pyplot as plt

import torch
import torch.nn               as nn
import torch.optim            as optim
import torchvision
import torchvision.models     as models
import torchvision.transforms as transforms

from cgi import test
import numpy as np

from scipy.signal import convolve2d, correlate2d
from skimage.util import random_noise
import torchvision
import random
import cv2
import torchvision.transforms as transforms
from utils import display_img, read_img
import PIL.Image as Image
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import read_img_mono, display_img, save_img, read_img, get_image_paths




f1_scores_SVM_csv_path = '../res/svm_f1.csv'
f1_scores_CNN_csv_path = '../res/cnn_f1.csv'

#Load purely numbers
data_svm = np.loadtxt(f1_scores_SVM_csv_path, delimiter=",")

data_cnn = np.loadtxt(f1_scores_CNN_csv_path, delimiter=",")

print(data_svm)
print(data_cnn)

perturbations = [
    'gaussian_pixel_noise',
    'gaussian_blurring',
    'image_contrast_increase', #DONE
    'image_contrast_decrease' , #DONE
    'image_brightness_increase' ,#DONE
    'image_brightness_decrease', #DONE
    'occlusion_image_increase', #DONE
    'salt_and_pepper' #DONE 
]
intensities = [0,1,2,3,4,5,6,7,8,9]
for ind, perturb in enumerate(perturbations):


    plt.title(perturb.capitalize())

    plt.xticks(intensities)
    plt.plot(intensities, data_svm[ind], label='SVM' ,marker = '.' )
    plt.plot(intensities, data_cnn[ind], label='CNN', marker = 'v')
    plt.legend()
    plt.xlabel('Intensities')
    plt.ylabel('F1 Score')
    
    plt.show()
    