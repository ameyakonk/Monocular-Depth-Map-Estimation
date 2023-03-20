# Monocular-Depth-Map-Estimation

## Overview
1. Implemented the paper, **High Quality Monocular Depth Estimation via Transfer Learning** to construct a depth map using a single image frame of a particular scene.
2. Incorporated augmentations into the image dataset to introduce and random horizontal flips during model training.
3. The model consisted of an UNet architecture with Densenet169 as a pre-trained encoder and a custom decoder.
4. Implemented a custom loss function incorporating edge loss, depth loss and Structural Similarity loss. 
5. Trained the model in PyTorch on a NYU2 Depth dataset and achieved an accuracy of 

## Personnel
### Ameya Konkar 

UID:118191058

Master's Student at University of Maryland, College Park

## Model
### Unet
UNet, evolved from the traditional convolutional neural network, was first designed and applied in 2015 to process biomedical images. As a general convolutional neural network focuses its task on image classification, where input is an image and output is one label, but in biomedical cases, it requires us not only to distinguish whether there is a disease, but also to localise the area of abnormality.
![unet](https://user-images.githubusercontent.com/78075049/226468500-b069cdc4-b6d2-493b-9ff7-b2580d1a67f2.png)

### Encoder
The densenet-169 model is used as encoder. It is one of the DenseNet group of models designed to perform image classification. The main difference with the densenet-121 model is the size and accuracy of the model.
![DenseNet-169](https://user-images.githubusercontent.com/78075049/226468124-afc0e1c2-47bd-4ec0-b79c-53113f1d68f0.png)

### Decoder 
The decoder used in the unet is constructed by upsampling the outputs w.r.t to the encoder.
