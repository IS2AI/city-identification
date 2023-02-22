# import all necessary libraries

import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from imutils import paths
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import seaborn as sns

# Setting the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

test_set = datasets.ImageFolder('preprocessed/patches/test_b', transform=t)
test_loader = DataLoader(test_set, batch_size=256, num_workers=4, shuffle=False)


model = torch.load('models/resnet50_v3/model_80.pth')
model.to(device)

print(test_set.class_to_idx)

len(test_loader.dataset)

testCorrect = 0
true_labels = []
pred_labels = []

# switch off autograd for evaluation
with torch.no_grad():

    # set the model in evaluation mode
    model.eval()

    # loop over the validation set
    for (x, y) in tqdm(test_loader):

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))

        # make the predictions and calculate the validation loss
        pred = model(x)
        
        true_labels += y.tolist()
        pred_labels += pred.argmax(1).tolist()

        # calculate the number of correct predictions
        testCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
testCorrect = testCorrect / len(test_loader.dataset)
print(testCorrect)

plt.figure(figsize=(30, 30))
confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
display_labels = ['ALA', 'ESB', 'ASB', 'NQZ', 'GYD', 'BKK', 'PEK', 'FRU', 'BOG', 'BOS', 'BNE', 'AEP', 'CAI', 'CHI', 'DUB', 'HAN', 'HKG', 'IST', 'CGK', 'FIH', 'KUL', 'LOS', 'LHE', 'LIS', 'MNL', 'MEL', 'MEX', 'MIL', 'BOM', 'MUC', 'NBO', 'OSL', 'PAR', 'RIX', 'SFO', 'GRU', 'ICN', 'CIT', 'SIN', 'SYD', 'TPE', 'TAS', 'TKY', 'YVR', 'IAD']

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['ALA', 'ESB', 'ASB', 'GYD', 'BKK', 'PEK', 'FRU', 'BOG', 'BOS', 'BNE', 'AEP', 'CAI', 'CHI', 'DUB', 'HAN', 'HKG', 'IST', 'CGK', 'FIH', 'LOS', 'LHE', 'LIS', 'MNL', 'MEL', 'MEX', 'MIL', 'BOM', 'MUC', 'NBO', 'NQZ', 'OSL', 'PAR', 'RIX', 'SFO', 'GRU', 'ICN', 'CIT', 'SIN', 'SYD', 'TPE', 'TAS', 'TKY', 'YVR', 'IAD'])


cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
group_counts = ["{0:0.1f}".format(value) for value in
                confusion_matrix.flatten()]
group_percentages = ["{0:.1%}".format(value) for value in
                     cm_normalized.flatten()]
labels = [f"{v2}\n{v3}" for v2, v3 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(45,45)

fig = sns.heatmap(confusion_matrix, annot=labels, fmt='',xticklabels=display_labels, yticklabels=display_labels, cmap='Blues')

plt.title('Confusion Matrix of City Classificaiton Model')
plt.xlabel('Predicted Label', fontsize = 15)
plt.ylabel('True Label', fontsize = 15)

plt.savefig('sns_conf_50.pdf')


inds = [x[0] for x in sorted(enumerate(group_percentages), key=lambda x: x[1])[-5:]]
for i in inds:
    t = (i % 44)
    print(display_labels[t])
    print(group_percentages[i])


f = open("resnet50_v3.txt", "w")
f.write('Accuracy: {}'.format(testCorrect))
f.close()


