# -*- coding: utf-8 -*-
# +
from config import *

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from sklearn.metrics import accuracy_score
# -

# # Data loading

trans = transforms.ToTensor() # PIL to torch.Tensor
dataset = ImageFolder(DATA_PATH, transform=trans)

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

# TODO : Ajouter le calcul de l’accuracy
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

def test(model, loss, test_loader, gpu):
    model.training = False
    with torch.no_grad():
        all_losses = []
        for inputs, targets in test_loader:
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


# ## Prédiction

def predict():
    pass
