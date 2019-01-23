# The basic usage of this file is that you'll pass in a single image /path/to/image and return the flower name and class probability.

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


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(path):
    checkpoint = torch.load(path)
    structure = checkpoint['model_algo']
    hidden_layer1 = checkpoint['hidden_layer_1']
    model_2,_,_ = model_setup(structure , 0.5,hidden_layer1)
    model_2.class_to_idx = checkpoint['class_to_idx']
    model_2.load_state_dict(checkpoint['state_dict'])

    return model_2


k = 5 # Default value of k
saved_checkpoint = None # Default value of saved checkpoint
data_dir = 'flowers/' # The default directory where we will train, val, and test dataset
train_dir = data_dir + 'train' # Default Variable to store the directory for train data
valid_dir = data_dir + 'valid' # Default Variable to store the directory for validation data
test_dir = data_dir + 'test' # Default Variable to store the directory for test data

gpu_available = None # Store the value of gpu being available

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

# Function to Process the test image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns the image as a tensor
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = transform(img)

    return tensor_image

# Default value of label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

parser = argparse.ArgumentParser()

# the two following two arguments are positional arguments
parser.add_argument("test_image_directory", help = "takes the path of the testing images.")
parser.add_argument("checkpoint", help = "load the previously trained model")

# The following three arguments are optional arguments
parser.add_argument("--top_k", type = int, help = "display the number of top k probabilities for a particular test image")
parser.add_argument("--category_names", help = "display the real category name of the flowers of the test data. takes input of the file with real names", type = str)
parser.add_argument("--gpu", help = "use gpu to compute to make prediction", action="store_true")

args = parser.parse_args()

test_image = data_dir + args.test_image_directory
saved_checkpoint = args.checkpoint


if args.top_k:
    k = args.top_k

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

if args.gpu:
    gpu_available = torch.cuda.is_available()


# Load the model from the saved checkpoint
model = load_model(saved_checkpoint)

# Map the index label
model.class_to_idx = train_data.class_to_idx
cat = model.class_to_idx

# Invert the mapping i.e index to class
idx_to_class = {v: k for k, v in model.class_to_idx.items()}

# Function to predict the flower name from an image along with the probability of that name
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    mod = model

    if gpu_available:
        mod.cuda()

    mod.eval()
    with torch.no_grad():
        out = mod.forward(img)

    prob = F.softmax(out.data,dim=1)

    return prob.topk(topk)

probs  = predict(image_path = test_image, model = model, topk = k)
print("#"*100)
print("""\t\t\t\t\t PREDICTION STARTED""")
print("#"*100)
print("\n")
print("The following are the top {} predicted Classes, their Name(s), their Probabilities:".format(k))
for i in range(len(np.array(probs[0][0]))):
    print(np.array(probs[1][0])[i], cat_to_name[str(np.array(probs[1][0])[i] + 1)],  np.array(probs[0][0])[i])
