# Image_Classification_Python
Link to original dataset: https://www.kaggle.com/datasets/hchen13/marvel-heroes

# Image Classification with Python3 
In this project, using 2 main methods, one classical, one with Deep Learning, we aim to classify marvel heros into 8 different categories (8 heros).
Methods:
1) Bag-of-Words Model using SIFT Descriptors and classficiation with Support Vector Machine (SVM)
2) Deep Learning using Pytorch
The dataset was resized before performing training and various augnmentations were conducted on the training set to increase our dataset size.


## Description

The dataset consists of 3035 images, each belonging to 1 of 8 classes. These were named based on a set of Marvel movie characters as follows : Black Widow, Captain America, Doctor Strange, Hulk, Ironman, Loki, Spider-man and Thanos. In the project, we aim to classify given images into one of these 8 categories by training 2 models. Firstly, this would be done with a Support Vector Machine (SVM) fitted on the images’ “Bag of Words”, generated with the SIFT interest point detector and descriptor. Secondly, this would be done with Deep Learning using a pretrained Convolutional Neural Network, namely ‘ResNet18’, and retraining it. The models will be trained using the provided training data (a folder containing folders for each of the classes mentioned above), then tuned using validation data (a subset of the training data). The models will then be tested on unseen data (images in a provided “test” folder) and have perturbations applied to the test images in order to see how well it generalises new data.


### Dataset and Input data
The images within the dataset used in the 2 models have been resized from the original dataset taken from Kaggle. This is to reduce the computational strain from working with large sized images. The image data has been split into 3 folders, namely; ‘Train’, ‘Valid’ and ‘Test’. Initially, the training data consists of roughly 300-400 images per category, totalling up to about 2500 images. However, in order to improve our classifier whilst working with limited data, we augment the data to increase the size of and variety within our dataset. For data augmentation, we used 4 types of transformations:
- Colour (brightness, contrast, saturation, hue) a.k.a Colour Jitter - Rotation
- Horizontal flip
- Vertical flip
Each augmented image could be affected by a mix of the above transformations. These images are then added to the training data to be used on the models. Our final training dataset consists of 4800 total images. The algorithms involved to create the image transformations are a mix of transformations from the pytorch library (please refer to augment.py in the Appendix)
The valid dataset is used to validate our trained models in order to aid us in tweaking our methodology.

### Testing
For testing, we are required to perturb the data in the following 8 ways:
1) Gaussian pixel noise
2) Gaussian blurring
3) Image contrast increase
4) Image contrast decrease
5) Image brightness increase
6) Image brightness decrease
7) Occlusion of the image increase
8) Salt and Pepper Noise
These perturbations are functions written in Python in the python file robustnessTest.py
Each perturbation performs some changes to pixel values for each image file. Image files are not directly supplied but rather a filepath to the image folders containing these images are provided. The utility functions in utils.py then help to provide the image encoded as value matrices to the perturbation functions.




## SIFT Bag of Words - Support Vector Machine Classifier Model
Libraries used:
- Opencv
- Numpy
- Scikit Learn - Torch
General pipeline:
1) Detect Interest Points, Describe Patch
2) Create codebook using K-Means clustering to cluster descriptors
3) Create a “Bag of Words” with built codebook for training, validation and test sets
4) Train a SVM to do multi-class classification given the ‘bag of words’ for images in the
training set
In the classical model we first use SIFT from the opencv library to first define the interest points, then compute the descriptors to describe the patch around the detected interest points. The opencv SIFT method uses a difference of gaussians blob filters across multiple scales to determine the interest points. These interest points are then defined by a histogram of gradients, by splitting each patch into a 4 x 4 cell-grid, each cell containing a histogram of gradients within the cell in 8 different directions. In total, we have a 4 x 4 x 8 = 128 dimensional vector describing each patch. Now, given all the patches in the data set, we build a codebook using the K-Means algorithm to cluster the 128-dimensional vectors, returning a list of cluster centres for the codebook. Here we chose to use 15 cluster centres for clustering. We then generate a Bag-of-Visual-Words for each image by quantizing each image using the same SIFT-HoG descriptor from opencv and determining the closest cluster centre for each image’s interest point patch descriptor and we count the matches per cluster into a histogram, i.e a ‘bag of words’. After which, with the SVM from Scikit Learn we fit and train the SVM using the ‘bag of words’ for the training images and their corresponding labels.

## Deep Learning CNN Classifier Model
Libraries used:
- Numpy
- Torch General pipeline:
1) Normalise and load data
2) Retrain the pretrained CNN model
a) Use mini-batching and mini-batch descent algorithm
b) Save the model returning best score
We use a pretrained CNN model in order to avoid having to train the model from scratch. For this we use the ResNet18 model from Pytorch, which is trained on 14 000 000 images and 1 000 image categories.
Our process of retraining the model starts by normalising the dataset. This is done as it generally leads to faster convergence and speeds up the Neural Network learning. Following this we pre-load the data into mini-batches of 32. This is crucial as it allows the model to scale for huge amounts of data. Instead of processing images individually, a mini-batch groups sets of images allowing them to be processed in parallel. This data is then supplied into a training function.
The training function takes in the following parameters:
1) ‘model’ - namely the pretrained Resnet18 model in this case.
2) ‘train_loader’ - the training data loaded into mini-batches.
3) ‘test_loader’ - the validation data loaded into mini-batches.
4) ‘criterion’ - namely the nn.CrossEntropyLoss() function in this case.
5) ‘optimiser’ - namely SGD in this case.
6) ‘n_epochs’ - the number of times the method iterates through the data.
As stated before, the function takes in the mini-batched ‘loaders’ in order to increase the processing efficiency. We supply the criterion parameter with the nn.CrossEntropyLoss() function as it is a function more punishing when incorrect predictions are made with high confidence. This in turn gives a more accurate F1 score. Through providing the SGD optimiser, we implement a stochastic gradient descent, an iterative method for optimising an objective function which reduces the computational burden. We also chose a low learning rate, which could aid in accuracy and hopefully attain the optimal.
The training function loops through all the data ‘n_epochs’ amount of times, each time running the images through the pretrained model and punishing for wrongly classified images using the criterion.



### Dependencies
Joblib python3
Pytorch
OpenCV


### Executing program
The models have been left out of the github project due to its large size. 




## Authors

Contributors names and contact info
Ezekiel Ang: https://www.linkedin.com/in/ezekielangjh/

