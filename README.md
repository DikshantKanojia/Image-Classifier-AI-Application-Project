# Image-Classifier-AI-Application-Project
## Description:
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, we might want to include an image classifier in a smart phone app. To do this, we will use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
In this project, we'll train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells us the name of the flower our camera is looking at. In practice we will train this classifier, then export it for use in our application. We'll be using the dataset of 102 flower categories


**The project is broken down into multiple steps:**
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

When we've completed this project, we will have an application that can be trained on any set of labeled images. Here our network will be learning about flowers and end up as a command line application.

## NOTE: 
THE 'checkpoint.pth' file and the datasets were too large to upload. 

## Technical uses
This project uses the following technical concepts, librarires, and softwares:

- Python
- NumPy
- pandas
- scikit-learn 
- Matplotlib
- tensorflow
- Deep Learning 
- Neural Networks : Vgg, Densenet, Alexnet


## Summary of results and process:
- Performed torchvision transformations to augment the training data with random scaling, rotations, mirroring, and cropping in order to effectively train the model.
- The training, validation, and testing data is appropriately cropped and normalized
- Implemented deep learning algorithms such as Vgg, Densenet, and alexnet to train classifier on a dataset with 102 categories of flowers.
- A new feedforward network is defined for use as a classifier using the features as input
- Achieved a validation accuracy of 88% and a test data accuracy of 86%
- Built a python application to be run directly from the terminal in order to classify images.
- Provided user-input functionalities such as GPU computation and transfer learning  in order for users to make customizations in the model during the training process.


