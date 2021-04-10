# -*- coding: utf-8 -*-
# # EDA (Exploratory Data Analysis)

# +
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from config import *
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
# -

# # Data loading

trans = transforms.ToTensor() # PIL to torch.Tensor
dataset = ImageFolder(DATA_PATH, transform=trans)

dataset

dataset.class_to_idx

# C’est quand même mieux d’avoir l’association label --> string

idx_to_class = {}
for key, val in dataset.class_to_idx.items():
    idx_to_class[val] = key

idx_to_class

# **On observe quelques images d’exemples**

indices = torch.randint(low=0, high=len(dataset), size=(10,))
subset = Subset(dataset, indices)

# +
fig, axs = plt.subplots(2, 5, figsize=(50, 20))

ind = 0
for i in range(2):
    for j in range(5):
        img, tgt = subset[ind]
        img = img.squeeze().permute(1, 2, 0)
        axs[i, j].imshow(img)
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        ind += 1
# -

# Excellent ! 🙌

# # Balancement du dataset

# +
all_targets = []
for _, targ in dataset:
    all_targets.append(targ)

all_targets = torch.tensor(all_targets)
# -

# %matplotlib inline
plt.figure(figsize=(20, 10))
plt.hist(all_targets.numpy())
plt.xticks([0, 1]);

# Fully balanced 👌

# # Baseline : Resnet-18

model = resnet18(pretrained=True)

# On connaît notre pattern :  
# 1. On supprime la dernière couche linéaire de classification.
# 2. On gèle les couches
# 3. On ajoute la dernière couche linéaire avec le bon nombre de classes (ici 2).

model

# La dernière couche du modèle (de classification), est nommée 'fc', c’est donc cette dernière que nous allons modifier et faire réapprendre.

model.fc = nn.Linear(in_features=512, out_features=1)

# 👍
#
# On va régler ce problème sous forme de régression logistique. On va prédire le pourcentage de chance d’appartenir à la classe 1, soit la classe *screenshot*.

model

for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False


# Excellent ! Nos couches sont gelées ! 🙌

# # Fonctions nécessaires

# ## Entrainement

# TODO : Implémenter le score (accuracy, matrice de confusion)
def train(model, optimizer, loss, train_loader, epochs=100, scheduler=None, valid_loader=None, gpu=None):
    # GPU
    if gpu:
        model = model.cuda(gpu)
    
    epochs_train_loss = []
    epochs_valid_loss = []
    for ep in range(epochs):
        
        all_losses = []
        for inputs, targets in train_loader:
            # GPU
            if gpu:
                inputs = inputs.cuda(gpu)
                targets = targets.cuda(gpu)
            
            predictions = model(inputs)
            err = loss(predictions, targets)
            
            # Machine is learning
            err.backward()
            optimizer.step()
            
            # Clean GPU
            if gpu:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                torch.cuda.empty_cache()
            
            all_losses.append(err.detach().cpu())
        
        # Validation step
        if valid_loader is not None:
            valid_losses = valid(model, loss, valid_loader, gpu)
            print(f'\r', end='')
        else:
            # Afficher les informations de l’époque
            print(f'\r', end='')


# ## Validation

def valid(model, loss, valid_loader, gpu):
    model.training = False
    with torch.no_grad():
        all_losses = []
        for inputs, targets in valid_loader:
            if gpu:
                inputs = inputs.cuda(gpu)
                targets = targets.cuda(gpu)
            
            predictions = model(inputs)
            err = loss(predictions, targets)
            
            all_losses.append(err.detach().cpu())
            
            # Clean GPU
            if gpu:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                torch.cuda.empty_cache()
        all_losses = torch.vstack(all_losses)
        return all_losses.mean()


# ## Test

def test():
    pass


# ## Prédiction

def predict():
    pass








