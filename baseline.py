# -*- coding: utf-8 -*-
# +
from config import *

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

import numpy as np
# -

# # Data loading

# +
# À définir manuellement
train_rate = 0.8
valid_rate = 0.1
batch_size = 4

trans = transforms.Compose([
    transforms.Resize(size=(1500, 1500)),
    transforms.ToTensor()
])
dataset = ImageFolder(DATA_PATH, transform=trans)

train_size = int(train_rate * len(dataset))
valid_size = int(valid_rate * len(dataset))

# On split avec train-valid-test = 80-10-10
all_indexes = np.arange(len(dataset))
train_indexes = np.random.choice(all_indexes, size=train_size, replace=False)
all_indexes = np.setdiff1d(all_indexes, train_indexes)
valid_indexes = np.random.choice(all_indexes, size=valid_size, replace=False)
test_indexes = np.setdiff1d(all_indexes, valid_indexes)

train_subset = Subset(dataset, train_indexes)
valid_subset = Subset(dataset, valid_indexes)
test_subset = Subset(dataset, test_indexes)

# Les dataloaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
# -

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

def train(model, optimizer, loss, train_loader, epochs=100, scheduler=None, valid_loader=None, gpu=None):
    # GPU
    if gpu is not None:
        model = model.cuda(gpu)

    epochs_train_loss = []
    epochs_valid_loss = []
    epochs_train_acc = []
    epochs_valid_acc = []
    for ep in range(epochs):
        model.training = True
        
        all_losses = []
        all_predictions = []
        all_targets = []
        for i, (inputs, targets) in enumerate(train_loader):
            # GPU
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)
            
            predictions = model(inputs).squeeze()
            err = loss(predictions, targets)

            # Machine is learning
            err.backward()
            optimizer.step()

            # Clean GPU
            if gpu is not None:
                err = err.detach().cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                torch.cuda.empty_cache()
            
            all_losses.append(err)
            all_predictions.append((predictions >= 0.5) * 1)
            all_targets.append(targets)
            
            print(f'\rBatch : {i+1} / {len(train_loader)}', end='')
        
        all_predictions = torch.hstack(all_predictions)
        all_targets = torch.hstack(all_targets)
        
        train_loss = np.hstack(all_losses).mean()
        train_acc = accuracy_score(all_targets, all_predictions)
        
        # Historique
        epochs_train_acc.append(train_acc)
        epochs_train_loss.append(train_loss)
        
        if scheduler is not None:
            scheduler.step()
        
        # Validation step
        if valid_loader is not None:
            valid_loss, valid_acc = valid(model, loss, valid_loader, gpu)
            # Historique
            epochs_valid_acc.append(valid_acc)
            epochs_valid_loss.append(valid_loss)
            print(f'\rEpoch : {ep+1} - Train Accuracy : {train_acc*100:.2f}% - Train Loss : {train_loss:.2e} - Valid Accuracy : {valid_acc*100:.2f}% - Valid Loss : {valid_loss:.2e}')
        else:
            # Afficher les informations de l’époque
            print(f'\rEpoch : {ep+1} - Train Accuracy : {train_acc*100:.2f}%  - Train Loss : {train_loss:.2e}')
        
    if valid_loader is not None:
        return epochs_train_acc, epochs_train_loss, epochs_valid_acc, epochs_valid_loss
    
    return epochs_train_acc, epochs_train_loss


# ## Validation

def valid(model, loss, valid_loader, gpu):
    model.training = False
    with torch.no_grad():
        all_losses = []
        all_predictions = []
        all_targets = []
        for inputs, targets in valid_loader:
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze()
            err = loss(predictions, targets)

            all_losses.append(err.detach().cpu())

            # Clean GPU
            if gpu is not None:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                predictions = predictions.cpu()
                torch.cuda.empty_cache()
                
            all_predictions.append((predictions >= 0.5) * 1)
            all_targets.append(targets)
        
        all_losses = torch.hstack(all_losses)
        all_predictions = torch.hstack(all_predictions)
        all_targets = torch.hstack(all_targets)
        valid_acc = accuracy_score(all_targets, all_predictions)
        
        return all_losses.mean(), valid_acc


# ## Test

def test(model, loss, test_loader, gpu):
    model.training = False
    with torch.no_grad():
        all_losses = []
        all_predictions = []
        all_targets = []
        for inputs, targets in test_loader:
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.float().cuda(gpu)

            predictions = model(inputs).squeeze()
            err = loss(predictions, targets)

            all_losses.append(err.detach().cpu())

            # Clean GPU
            if gpu is not None:
                err = err.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()
                torch.cuda.empty_cache()
                
            all_predictions.append((predictions >= 0.5) * 1)
            all_targets.append(targets)
            
        all_losses = torch.vstack(all_losses)
        all_predictions = torch.hstack(all_predictions)
        all_targets = torch.hstack(all_targets)
        test_acc = accuracy_score(all_targets, all_predictions)
        
        return all_losses.mean(), test_acc


# ## Prédiction

# Input dim : (batch_size, chanels, height, width)
def predict(model, tensor_data, gpu):
    model.training = False
    
    if gpu is not None:
        model = model.cuda(gpu)
        tensor_data = tensor_data.cuda(gpu)
    
    with torch.no_grad():
        predictions = model(tensor_data).squeeze()
    return (predictions >= 0.5) * 1


# # Entrainement

# +
learning_rate = 0.01

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.BCEWithLogitsLoss()

# Test avec le test set (bizarre me direz vous ...)
hist = train(model, optimizer, loss, train_loader=test_loader, epochs=100, gpu=0)
# -

# **Le réseau ne semble pas apprendre ...**


