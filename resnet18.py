from cgi import test
import os
from pyexpat import model
import numpy as np
import PIL.Image as Image
from   matplotlib import image, pyplot as plt

import torch
import torch.nn               as nn
import torch.optim            as optim
import torchvision
import torchvision.models     as models
import torchvision.transforms as transforms


classes = [
    "black widow",
    "captain america",
    "doctor strange",
    "hulk",
    "ironman",
    "loki",
    "spider-man",
    "thanos"
]


# Standard field
BATCH_SIZE = 32


# Folder paths
training_dataset_path = './data/train/'
test_dataset_path = './data/valid/'


# Need to resize for all images to be of the same size in order to be compared
training_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
# Declare the dataset for training and load it
train_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_transforms)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Calculate the mean and standard deviation of the dataset
# Used for normalisation, which leasds to faster convergience and speeds up NN learning
def get_mean_and_std(loader): 
    mean = 0
    std = 0
    total_images_count = 0
    for images,_ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std


mean,std = get_mean_and_std(train_loader)
print(mean, std)


# Define more transforms to make datasets less biased. ALL images have to be same size
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.25),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


# Declare datasets with new transforms
train_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root = test_dataset_path, transform = test_transforms)


# Function to show transformed images
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()
    print('labels: ', labels)

# Calls function to show images
#show_transformed_images(train_dataset)


# Load datasets based on the new transformations
train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False)


# Function deciding if to use GPU or CPU as GPU would be faster but is not always available
def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else: 
        dev = "cpu"
    return torch.device(dev)


# Function to train the neural network. This is the MAIN function.
def train_nn(model, train_loader, test_loader, criterion, optimiser, n_epochs):
    device = set_device()
    best_acc = 0  #keep track of best accuracy

    # If a model has been saved before, use that one
    if os.path.exists('./best_model.pth'):
        print('Old model found')
        model = torch.load('best_model.pth')

    # Training loop
    for epoch in range(n_epochs):
        print("Epoch number %d" % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0) 

            optimiser.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimiser.step()

            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct / total

        print("    - Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
            % (running_correct, total, epoch_acc, epoch_loss))

        test_dataset_acc = evaluate_model_on_test_set(model, test_loader)

        if (test_dataset_acc > best_acc):
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch, optimiser, best_acc)

    print("Finished")
    return model


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0) 

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            
            predicted_correctly_on_epoch += (predicted==labels).sum().item()

    epoch_acc = 100 * predicted_correctly_on_epoch / total
    print("    - Testing dataset. Got %d out of %d images correctly (%.3f%%)"
        % (predicted_correctly_on_epoch, total, epoch_acc))

    return epoch_acc


# Function used to keep track of the best epoch when training
def save_checkpoint(model, epoch, optimiser, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best accuracy': best_acc,
        'optimiser': optimiser.state_dict()
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')


# Set up parameters in order to call the train method
resnet18_model = models.resnet18(pretrained=True)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 8
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)


# Call the train method
train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimiser, n_epochs = 5) 


# Save the best model to a .pth file
checkpoint = torch.load('model_best_checkpoint.pth.tar')
resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 8
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
resnet18_model.load_state_dict(checkpoint['model'])

torch.save(resnet18_model, 'best_model.pth')



# Set up image transformations in order to be classified
image_transformations = transforms.Compose([ 
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


# Function used to test assigning an image into one of the classes
def classify(model, image_transforms, classes, image_path):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(classes[predicted.item()])


#model = torch.load('best_model.pth')
#classify(model, image_transformations, classes, "../190916093426-02-mark-ruffalo-hulk.jpg")