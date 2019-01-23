# Import the necessary libraries
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import torchvision.models as models

from collections import OrderedDict
from PIL import Image

import json
import argparse

# predefined default variables
mod = "densenet121"
lr = 0.001 # Default Learning rate
hidden_layer1 = 150 # Default hidden layer 1
epoch = 1 # Default epoch value
save_checkpoint_dir = 'checkpoint.pth'# Variable to store the directory to save the checkpoint
data_dir = 'flowers' # The default directory where we will train, val, and test dataset
train_dir = data_dir + '/train' # Default Variable to store the directory for train data
valid_dir = data_dir + '/valid' # Default Variable to store the directory for validation data
test_dir = data_dir + '/test' # Default Variable to store the directory for test data
gpu_available = None # Store the value of gpu being available

# List of deep learning pre-trained model
model_algos = {"vgg16": 25088,
               "alexnet": 9216 ,
               "densenet121": 1024}

# Define a function to create a model, optimizer, criterion
def model_setup(mod = "densenet121", dropout = 0.5, hidden_layer1 = 150, lr = 0.001):
    # lr -> learning rate
    # mod -> select the deep learning pre-trained model you want to use

    print("Model Setup Started... ")
    if mod == 'vgg16':
        model = models.vgg16(pretrained = True)

    elif mod == 'alexnet':
        model = models.alexnet(pretrained = True)

    elif mod == 'densenet121':
        model = models.densenet121(pretrained = True)

    else:
        print(f""" {mod} is not  a valid model. Please select from the following model:
                1. vgg16
                2. alexnet
                3. densenet121""")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('dropout', nn.Dropout(dropout)),
                                ('input', nn.Linear(model_algos[mod], hidden_layer1)),
                                ('Act_1: relu', nn.ReLU()),
                                ('Hidden_1', nn.Linear(hidden_layer1, 100)),
                                ('Act_2: relu', nn.ReLU()),
                                ('Hidden_2', nn.Linear(100, 60)),
                                ('Act_3: relu', nn.ReLU()),
                                ('Hidden_3', nn.Linear(60, 102)),
                                ('Output', nn.LogSoftmax(dim = 1))
                    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()

    if gpu_available:
        model.cuda()
        print("You are using GPU ")

    optimizer = optim.Adam(model.classifier.parameters(), lr )

    print("Model Setup Completed. ")

    return model, optimizer, criterion

# Functin to Train the model & check the accuracy on the validation dataset
def train_model(trainLoaders, model, optimizer, criterion, valLoaders,epochs = 1 ):

    print("#"*100)
    print("""\t\t\t\t\t STARTED TRAINING MODEL""")
    print("#"*100)
    print_every = 10
    steps = 0
    loss_display = []

    for e in range(epochs):
        running_loss = 0

        if gpu_available:
            model.cuda()

        for i, (inputs, labels) in enumerate(trainLoaders):
            steps += 1

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                for i, (inputs2, labels2) in enumerate(valLoaders):
                    optimizer.zero_grad()

                    with torch.no_grad():
                        outputs2 = model.forward(inputs2)
                        validation_loss = criterion(outputs2, labels2)
                        ps = torch.exp(outputs2).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                validation_loss = validation_loss / len(valLoaders)

                accuracy = accuracy / len(valLoaders)

                print("Epoch: {}/{}...".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Loss {:.4f}".format(validation_loss),
                          "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0





parser = argparse.ArgumentParser()
## Add the input arguments for train.py file
# The following argument is for the basic usage. It is a positional argument
parser.add_argument("train_data_directory", type = str,  help = "the data directory that contains the files to train the model")

# The following optional argument takes the data directory for validation
parser.add_argument("--valid_data_directory", type = str,  help = "the data directory that contains the files to validate the model")

# The following optional argument takes the data directory for testing
parser.add_argument("--test_data_directory", type = str, help = "the data directory that contains the files to test the model")

# The following optional argument to save the checkpoints after training the model
parser.add_argument("--save_dir", type = str, help = "Set directory to save checkpoints after training the model")
# The following optional argument is to choose the architecture or algorithm to train
parser.add_argument("--arch", help = "Choose the architecture/algorithm to train", choices =["vgg16" , "alexnet", "densenet121"] )

# The following are the optional arguments to set the hyperparameters
parser.add_argument("--learning_rate", type = float, help = "set the learning rate before training the model")
parser.add_argument("--hidden_units", type = int, help = "set the number of initial hiddent layer units")
parser.add_argument("--epochs", type = int, help = "number of times to train the model")

# The following optional argument is to set GPU usage if available
parser.add_argument("--gpu",  help = "If GPU is available use it to train the model", action="store_true")



args = parser.parse_args()

# Save the data_directory from which to train the data
train_dir = data_dir + args.train_data_directory

if args.valid_data_directory:
    valid_dir = data_dir + args.valid_data_directory

if args.test_data_directory:
    test_dir = data_dir + args.valid_data_directory

if args.save_dir:
    save_checkpoint_dir = args.save_dir

if args.arch:
    mod = args.arch

if args.learning_rate:
    lr = args.learning_rate

if args.hidden_units:
    hidden_layer1 = args.hidden_units

if args.epochs:
    epoch = args.epochs

if args.gpu:
    gpu_available = torch.cuda.is_available()



# Data Transforms for train set
data_transforms_train = transforms.Compose([transforms.RandomRotation(35),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])



# Data Transforms for validation set
data_transforms_validation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

# Data Transforms for test set
data_transforms_test = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
# For Training data
train_data = datasets.ImageFolder(train_dir, transform = data_transforms_train)

# For Validation Data
val_data = datasets.ImageFolder(valid_dir, transform = data_transforms_validation)

# For Test Data
test_data = datasets.ImageFolder(test_dir, transform = data_transforms_test)

# Define dataloaders for train data
trainLoaders = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)

# Define dataloaders for validation data
valLoaders = torch.utils.data.DataLoader(val_data, batch_size = 32, shuffle = True)

# Define dataloaders for test data
testLoaders = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True )

# Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Create the model, optimizer, and criterion variable
model, optimizer, criterion = model_setup(mod = mod, hidden_layer1 = hidden_layer1, lr = lr)

# Train the model
train_model(trainLoaders = trainLoaders, epochs = epoch, model = model, optimizer = optimizer, criterion = criterion, valLoaders = valLoaders)

#Function to  Save the checkpoint
# TODO: Save the checkpoint
model.class_to_idx = train_data.class_to_idx

def save_checkpoint():
    checkpoint = {'model_algo': 'densenet121',
                  'hidden_layer_1': 150,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')

# save the checkpoint of the above trained data
save_checkpoint()

## Additional Step to test the accuracy of the model on testing dataset
def testing_accuracy(testLoaders):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testLoaders:
            images3, labels3  = data
            output3 = model(images3)
            _, predicted = torch.max(output3.data, 1)
            total += labels3.size(0)
            correct += (predicted == labels3).sum().item()

    print("Accuracy for test data: %d %%" % (100*correct/total))

ask = input("""Do you want to test the accuracy of the model on unseen data:
                1. Yes
                2. No

            Enter the number of your choice """)

if ask == '1':
    testing_accuracy(testLoaders)
elif ask == '2':
    print("Okay!!! Bye Bye")
