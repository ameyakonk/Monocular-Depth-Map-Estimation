# Monocular-Depth-Map-Estimation

## Overview
1. Implemented the paper, **High Quality Monocular Depth Estimation via Transfer Learning** to construct a depth map using a single image frame of a particular scene.
2. Incorporated augmentations into the image dataset to introduce and random horizontal flips during model training.
3. The model consisted of an UNet architecture with Densenet169 as a pre-trained encoder and a custom decoder.
4. Implemented a custom loss function incorporating edge loss, depth loss and Structural Similarity loss. 
5. Trained the model in PyTorch on a NYU2 Depth dataset and achieved an accuracy of 70%

## Personnel
### Ameya Konkar 

UID:118191058

Master's Student at University of Maryland, College Park

## Results

Predicted output            | Ground Truth
:-------------------------:|:-------------------------:
![output1](https://user-images.githubusercontent.com/78075049/226758757-d53e1529-58fe-4ea1-8762-fee2c9fbac3b.png)  |  ![gt1](https://user-images.githubusercontent.com/78075049/226758776-d5bbff61-68be-4bbf-bceb-a47fcb2d5513.png)

Predicted output            | Ground Truth
:-------------------------:|:-------------------------:
![output2](https://user-images.githubusercontent.com/78075049/226758803-a921ac92-1c5d-44e6-ab2e-429644e63413.png)  | ![gt2](https://user-images.githubusercontent.com/78075049/226758833-4cf65616-69f5-4e8f-81dd-87399438b549.png)

## Model
### Unet
The **U-Net** architecture stems from the so-called “fully convolutional network” first proposed by Long, Shelhamer, and Darrell.

The main idea is to supplement a usual contracting network by successive layers, where pooling operations are replaced by upsampling operators. Hence these layers increase the resolution of the output. A successive convolutional layer can then learn to assemble a precise output based on this information.

One important modification in U-Net is that there are a large number of feature channels in the upsampling part, which allow the network to propagate context information to higher resolution layers. As a consequence, the expansive path is more or less symmetric to the contracting part, and yields a u-shaped architecture. The network only uses the valid part of each convolution without any fully connected layers. 
![unet](https://user-images.githubusercontent.com/78075049/226468500-b069cdc4-b6d2-493b-9ff7-b2580d1a67f2.png)

### Encoder
The **Densenet-169** model is used as encoder. It is one of the DenseNet group of models designed to perform image classification. The main difference with the densenet-121 model is the size and accuracy of the model. In the unet, the densenet is pretrained using **imagenet** weights. The final dense layer of the encoder is not used as classification was not the objective of the project.
![DenseNet-169](https://user-images.githubusercontent.com/78075049/226468124-afc0e1c2-47bd-4ec0-b79c-53113f1d68f0.png)

### Decoder 
The decoder used in the unet is constructed by upsampling the outputs w.r.t to the encoder.

### Loss function
The loss function consists of three kind of losses, edge loss, Structural similarity loss and overall depth loss between the predicted image and ground truth image.

## Dependencies
* Anaconda (Optional but recommended)
* Python3
* Matplotlib
* Pandas
* PyTorch (1.13.1 is used for development) 
* CUDA(optional)

## Run instructions
### Train model
'''
cd <repository_name>/src
python3 unet_train.py

### Test model
'''
cd <repository_name>/src
python3 unet_test.py



