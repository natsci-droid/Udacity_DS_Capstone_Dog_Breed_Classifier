# Udacity_DS_Capstone_Dog_Breed_Classifier

[image0]: https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier/blob/main/sample_images/Dog3.jpg "Dog"

![Dog][image0]


## Project Overview
Advances in Computer Vision mean that computers can now perform image recognition and classification on real images wth little preprocessing. This project uses a Convolutional Neural Network (CNN) to classify dog breeds if presented with an image of a dog, or identifies which dog breed a person most resembles if presented with a clear image of a face.

The data used are provided by Udacity, under the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) final capstone project. These include:
* dog images, split into train, validation and test partitions, provided by Udacity
* human images data set, provided by Udacity
* [ImageNet](http://www.image-net.org/), a heierarchical image database consisting of hundreds of images for each class of thousands of classes, thus very popular for training Convolutional Neural Networks.

The full project is written in the [Blog post](https://natsci-droid.github.io/Classifying-Dog-Breeds/).
The github repository is [here](https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier).

## Problem Statement
The goal is to build an algorithm that takes an image path, identifies whether a human or a dog is present, then classifies that image by dog breed. Three steps are required:
1) Determine if a human face is present
2) Determine if a dog is present
3) Classify the breed in the image

A separate classifier is required for each step, trained for the specific purpose, then must be combined for the overall app.


## Files
There are two data sets provided with this project, a set of dog images and a set of human images. Below are example images from this data set.

In total, there are 8351 dog images in 133 different breeds. 6680 are used for training, 835 for validation and 836 in the test set. There are 13233 human images to use to train the face detector.

<img src=https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier/blob/main/Brittany_02625.jpg  height="400"> <img src=https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier/blob/main/example_person.png  height="400">

### Repository
* dog_app.ipynb : notebook with classification app
* requirements.txt : library requirements
* extract_bottleneck_features.py : python code to extract model weights for training dog breed classifier using pretrained model
* saved_models : folder for best weights trained for the dog breed classifier
* sample_images: images used in final test
* 2021-01-26-Classifying Dog Breeds.md : blog post file

all other files are images for the blog, extracted from the notebook

## Instructions
The jupyter notebook contains the code used for the [blog post](https://natsci-droid.github.io/Classifying-Dog-Breeds/).

The extract_bottleneck_features.py script is required to extract features for retraining the dog classification model. This is not required to run the classifier algorithm, only to retrain.

To run the cells that retrain, extra data are required. These are not provided in the respository.

The trained weights of the final models are stored in the saved_models directory.

The notebook will classify either a dog or a person into a dog breed as the examples below.

Dog is a Dachshund  
<img src=https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier/blob/main/sample_images/Dog1.jpg  width="400">

Person resembles a Bull terrier  
<img src=https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier/blob/main/sample_images/Person3.jpg  width="400"> 

## Findings

The human face detector was validated against 100 sample human images and 100 sample dog images. 100% of human images were detected with a face, but also 11% of dog images. This means that dog images could be classified as human an incorrectly treated by the application. In order to avoid this, the human face detctor could be applied after the dog detector.

The dog detector was also validated against the same 100 sample human images and 100 sample dog images. Of the human images, a dog was detected in 0%, whereas a dog was detected in 100% of the dog data set. Given the higher performance over the human classifier, this should be used before the human classifier. The improved performance is likely to be due to the vast data set and the ability of deep learning to generalise on unseen data.

When trained from scratch the CNN achieves a classification accuracy of 1.1%.

When trained using the pre-trained VGG model, a classification accuracy of 42.7% is achieved.

The final dog breed classifier achieves an accuracy of 79.7% on the dog test data. The confusion matrix shows some confusion between classes, which could be the focus of new data collection.

<img src=https://github.com/natsci-droid/Udacity_DS_Capstone_Dog_Breed_Classifier/blob/main/cm.png  width="400">


## Requirements
opencv-python==3.2.0.6  
h5py==2.6.0  
matplotlib==2.0.0  
numpy==1.12.0  
scipy==0.18.1  
tqdm==4.11.2  
keras==2.0.2  
scikit-learn==0.18.1  
pillow==4.0.0  
ipykernel==4.6.1  
tensorflow==1.0.0  

##
Acknowledgements
Data ad project are provided by Udacity, with additional data from ImageNet and Unsplash.