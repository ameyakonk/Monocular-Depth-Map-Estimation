import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional import image_gradients
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchmetrics
import os

torch.cuda.empty_cache()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'[INFO] Using {device} for inference')

#######################################################################################################################################
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, base_path):
        self.img_labels = pd.read_csv(annotations_file)
        self.base_path = base_path

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.img_labels.iloc[idx, 0])
        label_path = os.path.join(self.base_path, self.img_labels.iloc[idx, 1])

        image = cv2.imread(img_path)
        label = cv2.imread(label_path)

        image = cv2.resize(image,(640, 480))
        label = cv2.resize(label, (320, 240))

        temp_transform = transforms.ToTensor()
        img_tr = temp_transform(image)
        label_tr = temp_transform(label)

        # calculate mean and std
        mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
        transform_norm_image = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        mean, std = label_tr.mean([1,2]), label_tr.std([1,2])
        transform_norm_label = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.Grayscale()])

        transform_flip = transforms.RandomHorizontalFlip(p=0.5)

        image = transform_norm_image(image)
        label = transform_norm_label(label)

        transform_flip(image)
        transform_flip(label)

        return image, label, img_path, label_path
    
###############################################################################################################################################
#
###############################################################################################################################################

test_file_name = "nyu2_test.csv"
train_file_name = "nyu2_train.csv"
model_name = "unet.pt"

batch_size = 1

# os.chdir("..")
# base_path = os.path.abspath(os.curdir)
base_path = "/home/ameya/Documents/Deep_Learning/Depth_estimation/"
img_dir = base_path + 'data/'
print(base_path)

train_file = base_path + 'csv/' + train_file_name
test_file = base_path + 'csv/' + test_file_name
model_path = base_path + 'models/' + model_name

train_dataset = CustomImageDataset(train_file, base_path)
test_dataset = CustomImageDataset(test_file, base_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("[INFO] Train dataloader size: ", len(train_dataloader), " ", "Batch size: ", batch_size)
print("[INFO] Train dataloader size: ", len(test_dataloader), " ", "Batch size: ", batch_size)

#########################################################################################################################################
#
#########################################################################################################################################

class UNet(nn.Module):
    def __init__(self, model):
        super(UNet, self).__init__()
        
        self.Densenet = model
        modules = list(self.Densenet.children())
        self.Densenet = nn.Sequential(*modules)[:-1]
       
        self.bneck_conv = nn.Conv2d(1664, 1664, kernel_size = (1,1), padding='same')

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.upsample1 = nn.Conv2d(1664, 832, kernel_size = (1,1), padding='same')
        self.upsample2 = nn.Conv2d(832, 416, kernel_size = (1,1), padding='same')
        self.upsample3 = nn.Conv2d(416, 192, kernel_size = (1,1), padding='same')
        self.upsample4 = nn.Conv2d(192, 64, kernel_size = (1,1), padding='same')

        self.upsample1_conv1 = nn.Conv2d(1664, 832, kernel_size = (1,1), padding='same')
        self.upsample1_conv2 = nn.Conv2d(832, 416, kernel_size = (1,1), padding='same')
        self.upsample1_conv3 = nn.Conv2d(384, 192, kernel_size = (1,1), padding='same')
        self.upsample1_conv4 = nn.Conv2d(128, 64, kernel_size = (1,1), padding='same')

        self.upsample2_conv1 = nn.Conv2d(832, 832, kernel_size = (1,1), padding='same')
        self.upsample2_conv2 = nn.Conv2d(416, 416, kernel_size = (1,1), padding='same')
        self.upsample2_conv3 = nn.Conv2d(192, 192, kernel_size = (1,1), padding='same')
        self.upsample2_conv4 = nn.Conv2d(64, 64, kernel_size = (1,1), padding='same')

        self.batchnorm1 = nn.BatchNorm2d(832)
        self.batchnorm2 = nn.BatchNorm2d(416)
        self.batchnorm3 = nn.BatchNorm2d(192)
        self.batchnorm4 = nn.BatchNorm2d(64)
        
        self.output = nn.Conv2d(64, 1, kernel_size = (3,3), padding='same')

        self.selected_out = {}
        self.fhooks = []
        # for i,l in enumerate(list(self.Densenet[0]._modules.keys())):
        #     print(l)
        #     if l in self.output_layers:
        #         print(getattr(self.Densenet[0],l))
        #         self.fhooks.append(getattr(self.Densenet[0],l).register_forward_hook(self.forward_hook(l)))

        self.fhooks.append(self.Densenet[0].denseblock3.denselayer19.relu1.register_forward_hook(self.forward_hook('denseblock3_19')))
        self.fhooks.append(self.Densenet[0].denseblock2.denselayer10.relu1.register_forward_hook(self.forward_hook('denseblock2_10')))
        self.fhooks.append(self.Densenet[0].denseblock1.denselayer5.relu1.register_forward_hook(self.forward_hook('denseblock1_5')))
        self.fhooks.append(self.Densenet[0].relu0.register_forward_hook(self.forward_hook('relu0')))

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    
    def upsampling(self, input_tensor, concat_layer, upsample, uc1, uc2, bn):
        x = self.upsample(input_tensor)
        x = upsample(x)
        x = torch.cat((x, concat_layer), 1)
        
        x = uc1(x)
        x = bn(x)
        x = uc2(x)
        x = bn(x)
        return x

    def forward(self, images):    
        dense_op = self.Densenet(images) 

        bneck = self.bneck_conv(dense_op)     
        x = nn.LeakyReLU(0.2)(bneck)

        x = self.upsampling(x, self.selected_out["denseblock3_19"], self.upsample1, self.upsample1_conv1, self.upsample2_conv1, self.batchnorm1)
        x = nn.LeakyReLU(0.2)(x)
        # shape [2, 832, 30, 40]

        x = self.upsampling(x, self.selected_out["denseblock2_10"], self.upsample2, self.upsample1_conv2, self.upsample2_conv2, self.batchnorm2)
        x = nn.LeakyReLU(0.2)(x)
        # shape [2, 416, 60, 80]

        x = self.upsampling(x, self.selected_out["denseblock1_5"], self.upsample3, self.upsample1_conv3, self.upsample2_conv3, self.batchnorm3)
        x = nn.LeakyReLU(0.2)(x)
        # shape [2, 192, 120, 160]
        
        x = self.upsampling(x, self.selected_out["relu0"], self.upsample4, self.upsample1_conv4, self.upsample2_conv4, self.batchnorm4)
        # shape [2, 64, 240, 320]

        x = self.output(x)
        return x    

##########################################################################################################################################
#
##########################################################################################################################################

def loss_fn(y_true, y_pred):
    l_depth = torch.mean(torch.abs(y_true - y_pred), axis=-1)
    dy_true, dx_true = image_gradients(y_true)
    dy_pred, dx_pred = image_gradients(y_pred)
    l_edges = torch.mean((torch.abs(dy_true - dy_pred) + torch.abs(dx_true - dx_pred)), axis=-1)

    l_ssim = torch.clip((1 - ssim( y_true, y_pred, data_range=1, size_average=True)) * 0.5, 0, 1)
    w1, w2, w3 = 1.0, 1.0, 0.1
    return (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth))
    
def accuracy_function(y_true, y_pred):
    y_true = torch.round(y_true).cpu().detach().numpy()
    y_pred = torch.round(y_pred).cpu().detach().numpy()
    
    return np.mean(np.equal(y_true, y_pred))
    # print(torch.equal(torch.round(y_true), torch.round(y_pred)))
    # return torch.mean(torch.equal(torch.round(y_true), torch.round(y_pred)))
    
##########################################################################################################################################
#
##########################################################################################################################################

encoder = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169')
model = UNet(encoder)
model = model.cuda() if device else model

best_accuracy = 0
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
epochs = 10

for t in range(epochs):
    print(f"[INFO] Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (X, y, _, _) in enumerate(train_dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        optimizer.zero_grad()
        loss = loss_fn(y, pred)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"[INFO] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    size = len(test_dataloader.dataset)
    accuracy = 0
    with torch.no_grad():
        print("[INFO] Testing model...")
        for (X_, y_, _, _) in test_dataloader:
            X_ = X_.to(device)
            y_ = y_.to(device)
            pred_ = model(X_)
            accuracy += accuracy_function(y_, pred_)
        
        print(f"[INFO] Accuracy: {accuracy/size}")
        if (accuracy/size*100) > best_accuracy:
            best_accuracy = (accuracy/size*100)
            torch.save(model.state_dict(), model_path)
            print("[INFO] Model Saved!!!")
        print()