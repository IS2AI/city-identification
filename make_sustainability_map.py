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

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets


# In[2]:


# read sustainability index csv

df = pd.read_csv('workspace/dataset/sustainability_index.csv', index_col='city')
df.dropna(subset=['overall'], inplace=True)
scores = df['overall'].to_dict()
scores


# In[3]:


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


# In[4]:


# setting the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# In[5]:


normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


# In[6]:


def interpolate_scores(A, out_size):
    """
    Function that interpolates input matrix A to a matrix of size out_size
    More info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    """
    h, w = A.shape
    vals = np.reshape(A, h*w)
    pts = np.array([[i,j] for i in np.linspace(0,1,w) for j in np.linspace(0,1,h)] )
    grid_x, grid_y = np.mgrid[0:1:out_size*1j, 0:1:out_size*1j]
    grid_z = inter.griddata(pts, vals, (grid_x, grid_y), method='linear')
    return grid_z


# In[7]:


def sustainability_map(image_path, true_index, model_path, device, patch_dim=480, save_dir='reg_results_diff'):
    """
    Divides image into patches and runs regression on each patch.
    Displays resulting mask of squares on input image.
    """
    plt.figure(figsize=(12, 12), dpi=300)
    plt.axis('off')

    # load the model
    model = torch.load(model_path)
    model.to(device)
    
    # load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape
    
    scores = np.zeros(((int(H/patch_dim)), (int(W/patch_dim))))     # stores difference between true and predicted indices  
    mask = np.zeros((H, W, 3))                                      # stores mask
    
    # iterate over pathes, predict and store the score
    i, j = 0, 0
    while (i+1)*patch_dim <= H:
        while (j+1)*patch_dim <= W:
            patch = image[i*patch_dim:(i+1)*patch_dim, j*patch_dim:(j+1)*patch_dim, :]
            x = normalize(patch)
            x = x.view((1, 3, 224, 224))
            x = x.to(device)
            pred = model(x).tolist()[0][0] * 100
            a = true_index - pred  
            scores[i, j] = a
            j += 1
        j = 0
        i += 1
        
    # mapping scores to colorpad
    norm = matplotlib.colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.RdYlBu)
    
    # color mask regions according to colorpad
    for i in range(int(H/patch_dim)):
        for j in range(int(W/patch_dim)):
            a = scores[i, j]
            mask[i*patch_dim:(i+1)*patch_dim, j*patch_dim:(j+1)*patch_dim, 0] = np.ones((patch_dim, patch_dim)) * mapper.to_rgba(a)[0]
            mask[i*patch_dim:(i+1)*patch_dim, j*patch_dim:(j+1)*patch_dim, 1] = np.ones((patch_dim, patch_dim)) * mapper.to_rgba(a)[1]
            mask[i*patch_dim:(i+1)*patch_dim, j*patch_dim:(j+1)*patch_dim, 2] = np.ones((patch_dim, patch_dim)) * mapper.to_rgba(a)[2]
            j += 1
        j = 0
        i += 1
    
    clb = plt.colorbar(mapper, orientation='horizontal', fraction=0.046, pad=0.04)
    clb.ax.tick_params(labelsize=6) 
    clb.ax.set_title('true ({}) - predicted'.format(true_index), fontsize=8)
    
    image_name = image_path.split('/')[-1].split('.')[0]

    plt.imshow(image) 
    plt.imshow(mask, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, image_name + '_overlay.jpg'))
    #plt.savefig(save_path)


# In[17]:


def sustainability_map_cnn(image_path, true_index, model_path, device, stride=60, patch_dim=480, save_dir='reg_results_overall'):
    plt.figure(figsize=(12, 12), dpi=300)
    plt.axis('off')

    print('[INFO] Load model and image')
    model = torch.load(model_path)
    model.to(device)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    S_in = image.shape[0]
    S_out = int((S_in-patch_dim)/stride) + 1
    
    scores = np.zeros((S_out, S_out))        
    mask = np.zeros((S_in, S_in, 3))
    
    print('[INFO] Compute predictions')
    start = time.time()
    i, j = 0, 0
    k, l = 0, 0
    while i+patch_dim <= S_in:
        while j+patch_dim <= S_in:
            patch = image[i:i + patch_dim, j:j + patch_dim, :]
            x = normalize(patch)
            x = x.view((1, 3, 224, 224))
            x = x.to(device)
            pred = model(x).tolist()[0][0] * 100
            scores[k, l] = true_index - pred
            j += stride
            l += 1
        j = 0
        l = 0
        i += stride
        k += 1
       
    end = time.time()
    print("[TIME] predictions computed in {:.2f}s".format(end - start))
    
    print('[INFO] Interpolate predictions')
    start = time.time()

    inter_scores = interpolate_scores(scores, S_in)
    
    end = time.time()
    print("[TIME] interpolation completed in {:.2f}s".format(end - start))

    norm = matplotlib.colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.RdYlBu)
    
    print('[INFO] Compute the mask')
    start = time.time()
    
    for i in range(S_in):
        for j in range(S_in):
            mask[i, j, :] = mapper.to_rgba(inter_scores[i, j])[:-1]

    end = time.time()
    print("[TIME] mask computed in {:.2f}s".format(end - start))
    
    clb = plt.colorbar(mapper, orientation='horizontal', fraction=0.046, pad=0.04)
    clb.ax.tick_params(labelsize=6) 
    clb.ax.set_title('predicted index', fontsize=8)


    if true_index == 0:
        true_index = 'unknown'
    plt.title('Difference of actual and predicted indexes for city with true index {}'.format(true_index))
    
    image_name = image_path.split('/')[-1].split('.')[0]
    
    # save only mask
    plt.imshow(mask)
    plt.savefig(os.path.join(save_dir, image_name + '_mask.jpg'))
    
    # save city and mask
    plt.imshow(image)
    plt.imshow(mask, alpha=0.3)
    plt.savefig(os.path.join(save_dir, image_name + '_overlay.jpg'))


# In[ ]:


model_path = 'models/resnet50_reg_overall/model_14.pth'

image_paths = list(paths.list_files('workspace/dataset/samples', validExts='jpg'))

# iterate over images
for image_path in image_paths:
    print(image_path)
    image_name = os.path.basename(image_path)
    print(image_name)
    code = image_name.split('_')[0]
    true_index = 0
    
    # get true sustainability index
    for key in scores.keys():
        key_code = key.split('(')[-1][:]
    
        if code == key_code:
            true_index = int(scores[key])
            print(true_index)
      
    # compute map
    sustainability_map_cnn(image_path, true_index, model_path, device)
    #sustainability_map(image_path, true_index, model_path, device)

# In[ ]:





# In[1]:


import shutil
shutil.make_archive('reg_results_overall', 'zip', 'reg_results_overall')


# In[ ]:




