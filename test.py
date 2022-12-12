from cgi import test
import joblib
import numpy as np
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from svm import bag_of_words



import os
import glob



# test every perturbation in every intensity on both SVM and ResNet CNN
# Save values to CSV file




'''
SVM TESTING -- 
'''
# #SVM
# Standard field
BATCH_SIZE = 10
DATA_PATH = 'data'
IMAGE_CATEGORIES = [
    'black widow', 'captain america', 'doctor strange', 'hulk', 'ironman', 'loki', 'spider-man', 'thanos'
]
SIFT_MAX_FEATURES = 50
CODEBOOK_FILE = 'codebook.joblib'
SVM_MODEL_FILE = 'svm_bow_mode.joblib'

perturbations = [
    # 'default',
    'gaussian_pixel_noise',
    'gaussian_blurring',
    'image_contrast_increase', #DONE
    'image_contrast_decrease' , #DONE
    'image_brightness_increase' ,#DONE
    'image_brightness_decrease', #DONE
    'occlusion_image_increase', #DONE
    'salt_and_pepper' #DONE
    
]


### ----- SVM and CNN ----
'''
Tests svm across all perturbations and intensities and returns a matrix
'''
def test_svm():
    #Loading Codebook
    if os.path.exists(CODEBOOK_FILE):
        print("Loading Codebook")
        codebook = joblib.load(CODEBOOK_FILE)
    else:
        print('no codebook')

    scaler = StandardScaler()

    f1_results_svm = np.zeros((8,10))
    #Test SVM 
    for ind,each in enumerate(perturbations):
        for intensity in range(10):
            perturbation_name = each

            test_img_paths, test_labels = get_test_image_paths(DATA_PATH, IMAGE_CATEGORIES, 6, perturbation_name, intensity)

            #SVM Testing
            print('Generating BoW features for test set: ' + perturbation_name + ' Intensity: ' + str(intensity))
            test_images = bag_of_words(test_img_paths, codebook)
            test_images_scaled = scaler.fit_transform(test_images)
            print('Test images:', test_images.shape)



            if os.path.exists(SVM_MODEL_FILE):
                print('Loading existing linear SVM model...')
                svm = joblib.load(SVM_MODEL_FILE)
            else:
                print('No linear SVM...')
                
            test_predictions = svm.predict(test_images_scaled)
            accuracy = accuracy_score(test_labels, test_predictions)
            print('Classification accuracy of SVM with BOW features: ', accuracy)

            #F1 score F1 = 2 * (precision * recall) / (precision + recall)
            #F1 score can be interpreted as a harmonic mean of the precision and recall, 
            #where an F1 score reaches its best value at 1 and worst score at 0
            f1 = 100 * f1_score(test_labels, test_predictions, average='macro')
            print(perturbation_name + ' ' + str(intensity))
            print('Classification f1 score(macro averaged) of SVM with BOW features: ', f1)
            #Save to svm f1 results matrix
            f1_results_svm[ind][intensity] = f1

    #Save f1_results for SVM to CSV:
    return f1_results_svm

F1_RES_SVM = 'f1svm.joblib'

if os.path.exists(F1_RES_SVM):
    print('Loading existing linear SVM model...')
    f1_results_svm = joblib.load(F1_RES_SVM)
else:
    print('testing svm...')
    f1_results_svm = test_svm()
    joblib.dump(f1_results_svm, F1_RES_SVM)
    print('Tested SVM Model')

np.savetxt('../res/svm_f1.csv', f1_results_svm, delimiter=',')


### -----  Start Deep Learning CNN Tests ------ 

# Function deciding if to use GPU or CPU as GPU would be faster but is not always available
def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else: 
        dev = "cpu"
    return torch.device(dev)

'''
Takes in a trained model and a test_loader. Runs the test and outputs accuracy and f1score as a tuple
'''
def evaluate_model_on_test_set(model, test_loader):
    #Set to evaluate mode
    model.eval()

    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    predictions = []
    actual = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0) 
            # print(labels) ##tensor of labels
            # print(images)
            outputs = model(images)
            
            
            _, predicted = torch.max(outputs.data, 1)
            


            #true pos
            predicted_correctly_on_epoch += (predicted==labels).sum().item()

            predictions.extend(predicted.numpy())
            actual.extend(labels.numpy()) 

    #Accuracy in percentage
    epoch_acc = 100 * predicted_correctly_on_epoch / total
    print("    - Testing dataset. Got %d out of %d images correctly (%.3f%%)"
        % (predicted_correctly_on_epoch, total, epoch_acc))
    
    #F1-score
    F1_score = 100 * f1_score(actual, predictions, average='micro')
    print("    - Testing dataset. F1-Score: ", F1_score)

    return epoch_acc, F1_score

'''
Returns f1-score matrix for CNN
'''
def test_cnn(num_of_test_per_cat):
    print('Running CNN tests')
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    f1_results_cnn = np.zeros((8,10))
    model = torch.load('best_model.pth')
    for ind,each in enumerate(perturbations):
        for intensity in range(10):
            perturbation_name = each
            print(each + " intensity: " + str(intensity))
            
            test_img_paths = DATA_PATH + '/test/' + perturbation_name + '/' + str(intensity)
            # print(test_img_paths)
            test_dataset = torchvision.datasets.ImageFolder(root = test_img_paths, transform = test_transforms)
            test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=True)
            accuracy, f1_score = evaluate_model_on_test_set(model, test_loader)
            
            print("Accuracy: " + str(accuracy))
            print("F1Score: " + str(f1_score))
            f1_results_cnn[ind][intensity] = f1_score
    return f1_results_cnn


F1_RES_CNN = 'f1cnn.joblib'

if os.path.exists(F1_RES_CNN):
    print('Loading existing CNN model results...')
    cnn_res = joblib.load(F1_RES_CNN)
else:
    print('testing cnn...')
    cnn_res = test_cnn(10)
    joblib.dump(cnn_res, F1_RES_CNN)
    print('Tested cnn Model')

np.savetxt('../res/cnn_f1.csv', cnn_res, delimiter=',')


print("Created CSVs.")
