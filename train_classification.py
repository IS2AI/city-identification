# import all necessary libraries

import os
import cv2
import time
import wandb
import numpy as np
from tqdm import tqdm
from imutils import paths
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from collections import Counter

def make_dir(dirName):
    # Create a target directory & all intermediate 
    # directories if they don't exists
    
    if not os.path.exists(dirName):
        os.makedirs(dirName, exist_ok = True)
        print("[INFO] Directory ", dirName,  " created")
    else:
        print("[INFO] Directory ", dirName,  " already exists")


INIT_LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 100

t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


train_set = datasets.ImageFolder('preprocessed/patches/train', transform=t)
val_set = datasets.ImageFolder('preprocessed/patches/val', transform=t)
                    

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, sampler=ImbalancedDatasetSampler(train_set))
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

print(dict(Counter(train_set.targets)))
print(train_set.class_to_idx)

print(dict(Counter(val_set.targets)))
print(val_set.class_to_idx)


# Setting the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# calculate steps per epoch for training and validation set
trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(val_loader.dataset) // BATCH_SIZE
print(trainSteps, valSteps)


# initialize the ResNet model
print("[INFO] initializing the ResNet model...")
init_model = models.resnet50(pretrained=True)
init_model.fc = nn.Linear(init_model.fc.in_features, 45)
#model.to(device)
model = nn.DataParallel(init_model, device_ids=[0,1,2])
model.to(device)

# initialize our optimizer and loss function
opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()

print(model)


NAME = 'resnet50_v3'
wandb.init(project="resnet50_v3", name=NAME)
make_dir('models/{}'.format(NAME))
    
wandb.config = {
 "learning_rate": INIT_LR,
 "epochs": BATCH_SIZE,
 "batch_size": EPOCHS
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over epochs
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
    
    # loop over the training set
    for (x, y) in tqdm(train_loader):

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        
        # perform a forward pass and calculate the training loss
        pred = model(x)        
        loss = lossFn(pred, y)
        
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
    
    # switch off autograd for evaluation
    with torch.no_grad():
        
        # set the model in evaluation mode
        model.eval()
        
        # loop over the validation set
        for (x, y) in tqdm(val_loader):

            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            
            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
            
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(train_loader.dataset)
    valCorrect = valCorrect / len(val_loader.dataset)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect))

    with open("epochs_resnet50_v3.txt", "w") as file_object:
        file_object.write("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        file_object.write("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss.cpu().detach().numpy(), valCorrect))
        
    wandb.log({"Train loss": avgTrainLoss.cpu().detach().numpy(), "Train accuracy": trainCorrect,
              "Val loss": avgValLoss.cpu().detach().numpy(), "Val accuracy": valCorrect})
    
    # serialize the model to disk
    torch.save(model, 'models/{}/model_{}.pth'.format(NAME, e + 1))
    
# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

