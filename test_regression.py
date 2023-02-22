#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all necessary libraries

import os
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from imutils import paths
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import r2_score

# In[2]:


# read sustainability index csv

df = pd.read_csv('city_indexes/sustainability_index.csv', index_col='city')
df['overall'] /= 100               # overall / planet / people / profit
scores = df['overall'].to_dict()
scores


# In[3]:


class CityDataset(Dataset):
    """City dataset"""

    def __init__(self, image_dir, scores, transform=None):
        """
        Args:
            image_dir: a list containing paths to images in the set.
            scores: dictionary containing sustainability scores.
            transform: transform to be applied on loaded images.
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


# In[4]:


t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# In[5]:


# get paths to the images
test_paths = list(paths.list_files('preprocessed/patches/test', validExts='jpg'))

# load the dataset
test_set = CityDataset(test_paths, scores, transform=t)
test_loader = DataLoader(test_set, batch_size=256, num_workers=4, shuffle=False)


# In[6]:


# additional fully connected layer to ResNet for regression

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
#         self.fc1 = nn.Linear(512, 38)        
        self.fc2 = nn.Linear(38, 1)
    
    def forward(self, x):
#         x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        return x


# In[7]:


# setting the device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# In[8]:


# load weights
model = torch.load('models/resnet101_reg_overall/model_19.pth')
model.to(device)


# In[9]:


totalTestLoss = 0
trues = []
preds = []

lossFn = nn.MSELoss()
testSteps = len(test_loader.dataset) // 256

# switch off autograd for evaluation
with torch.no_grad():

    # set the model in evaluation mode
    model.eval()

    # load batches from test loader
    for (x, y) in tqdm(test_loader):
        
        # send the input to the device
        (x, y) = (x.to(device), y.to(device, dtype=torch.float32))
        y = y.view((-1, 1))
        
        # make the predictions and calculate the test loss
        pred = model(x)
        loss = lossFn(pred, y)
        
        totalTestLoss += loss
        
        # save to list
        trues += y.tolist()
        preds += pred.tolist()
        
avgTestLoss = totalTestLoss / testSteps
print(avgTestLoss)


# In[10]:


# map to 1-100 range
pred_labels = (np.array(preds) * 100)
true_labels = np.round(np.array(trues) * 100)
#print(math.modf(pred_labels)[0])
#pred_labels = np.where(np.vectorize(math.modf(pred_labels)[0])<0.5, np.floor(pred_labels), np.ceil(pred_labels))
r2 = r2_score(true_labels, pred_labels)
print('r2 score for the model is', r2)
df_labels = pd.DataFrame(list(zip(pred_labels, true_labels)),
               columns =['pred_labels', 'true_labels'])
df_labels.to_csv('overall.csv')
n = true_labels.size
n


# In[11]:


# RMSE function

def rmse(pred_labels, true_labels):
    return np.sqrt(np.sum((pred_labels-true_labels)**2) / n)


# In[12]:


print(rmse(pred_labels, true_labels))


# In[13]:


# get true score values present in the set and their count
values, counts = np.unique(true_labels, return_counts=True)
values, counts


# In[14]:


# get the array indexes for all cities
# assumption: scores are unique, thus used as a key

city_indexes = {}

for value in values:
    city_indexes[value] = np.where(true_labels == value)[0]


# In[15]:


city_indexes


# In[16]:


# get predictions for each city
city_predictions = {}

for key in city_indexes:
    city_predictions[key] = pred_labels[city_indexes[key]]
    
# get statistics for each city
city_means = {}
city_stds = {}

for key in city_indexes:
        city_means[key] = np.mean(city_predictions[key])
        city_stds[key] = np.std(city_predictions[key])


# In[17]:


# obtain city codes
city_codes = []

for key in city_means.keys():
    city_codes.append(df.index[df['overall'] == key / 100].tolist()[0])


# In[18]:


plt.figure(figsize=(17, 5), dpi=100)

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.grid(axis='y', linestyle='-', linewidth=0.5)
plt.margins(x=0.007)
csfont = {'fontname':'Arial'}

plt.errorbar(city_codes, city_means.values(), city_stds.values(), 
             linestyle='None', marker='o', ecolor='black', elinewidth=1, color='red', label='mean predicted index with std')
plt.scatter(city_codes, city_means.keys(), color='g', marker='D', label='true index')

plt.xlabel('Cities', **csfont)
plt.ylabel('Index 1 (best) to 100 (worst)', **csfont)
plt.title('Sustainability scores for the test set (overall)', **csfont)
plt.legend()
plt.savefig('index_overall_resnet101.png')



