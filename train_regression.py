#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all necessary libraries

import os
import cv2
import time
import wandb
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from imutils import paths
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler

import torch.nn.functional as F


# In[2]:


def make_dir(dirName):
    # Create a target directory & all intermediate 
    # directories if they don't exists
    
    if not os.path.exists(dirName):
        os.makedirs(dirName, exist_ok = True)
        print("[INFO] Directory " ,dirName,  " created")
    else:
        print("[INFO] Directory " ,dirName,  " already exists")


# In[3]:


INIT_LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 50


# In[4]:


df = pd.read_csv('city_indexes/sustainability_index.csv', index_col='city')
#df['city'] = df['city'].str.extract(r"\(([A-Za-z]+)\)", expand=False)
df['overall'] /= 100
scores = df['overall'].dropna().to_dict()
print(scores)


# In[5]:


class CityDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, scores, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with scores.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.scores = scores
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):

        img_name = self.image_dir[idx]
        image = Image.open(img_name)
        
        city = img_name.split('/')[5].split('_')[0]
        score = scores[city]

        if self.transform:
            image = self.transform(image)

        return image, score


# In[6]:


t = transforms.Compose([
#         transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# In[7]:


train_paths = list(paths.list_files('preprocessed/patches/train', validExts='jpg'))
val_paths = list(paths.list_files('preprocessed/patches/val', validExts='jpg'))
random.shuffle(train_paths)
random.shuffle(val_paths)


# In[8]:


train_set = CityDataset(train_paths, scores, transform=t)
val_set = CityDataset(val_paths, scores, transform=t)                  

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


# In[9]:


# Setting the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# In[10]:


# calculate steps per epoch for training and validation set
trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(val_loader.dataset) // BATCH_SIZE
print(trainSteps, valSteps)


# In[11]:


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
#         self.fc1 = nn.Linear(512, 38)        
        self.fc2 = nn.Linear(38, 1)
    
    def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x)) 
        x = self.fc2(x)
        
        return x


# In[15]:


# initialize the ResNet model
print("[INFO] initializing the ResNet model...")
model = torch.load('models/resnet50_v3/model_80.pth')

    
model.module.fc = nn.Linear(model.module.fc.in_features, 1)
model = model.to(device)


# initialize our optimizer and loss function
opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.MSELoss()


NAME = 'resnet50_reg_overall'
wandb.init(project="resnet50_reg_overall", name=NAME)
make_dir('models/{}'.format(NAME))
    
wandb.config = {
  "learning_rate": INIT_LR,
  "epochs": BATCH_SIZE,
  "batch_size": EPOCHS
}

# measure how long training is going to take
print("[INFO] training the network...")

# loop over our epochs
for e in range(0, EPOCHS):
    
    # set the model in training mode
    model.train()
    
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    
#     startTime = time.time()
    # loop over the training set
    for (x, y) in tqdm(train_loader):

        # send the input to the device
        (x, y) = (x.to(device), y.to(device, dtype=torch.float32))
        
        # perform a forward pass and calculate the training loss
        pred = model(x)        
        y = y.view((-1, 1))
        
#         print(y, pred)
        
        loss = lossFn(pred, y)
                
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        totalTrainLoss += loss
        
        # finish measuring how long training took
#         endTime = time.time()
#         print("[INFO] time taken to train on one batch: {:.2f}s".format(endTime - startTime))
    
    # switch off autograd for evaluation
    with torch.no_grad():
        
        # set the model in evaluation mode
        model.eval()
        
        # loop over the validation set
        for (x, y) in tqdm(val_loader):

            # send the input to the device
            (x, y) = (x.to(device), y.to(device, dtype=torch.float32))
            y = y.view((-1, 1))
            
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)
    
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    
#     scheduler.step(avgValLoss)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Val loss: {:.6f}".format(avgTrainLoss, avgValLoss))
        
    wandb.log({"Train loss": avgTrainLoss.cpu().detach().numpy(),
               "Val loss": avgValLoss.cpu().detach().numpy()})
    
    # serialize the model to disk
    torch.save(model, 'models/{}/model_{}.pth'.format(NAME, e + 1))




