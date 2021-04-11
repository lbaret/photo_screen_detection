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

import numpy as np

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

np.random.shuffle(train_indexes)
np.random.shuffle(valid_indexes)
np.random.shuffle(test_indexes)

train_subset = Subset(dataset, train_indexes)
valid_subset = Subset(dataset, valid_indexes)
test_subset = Subset(dataset, test_indexes)

# Les dataloaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# +
model = resnet18(pretrained=True)

# Remplacement de la dernière couche
model.fc = nn.Linear(in_features=512, out_features=1)

# Gel des couches
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

learning_rate = 0.1

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.BCEWithLogitsLoss()
# -

before_layer = next(model.fc.parameters()).clone().detach()

imgs, tgts = next(iter(test_loader))
tgts = tgts.float()

predictions = model(imgs).squeeze()

err = loss(predictions, tgts)

err.backward()
optimizer.step()

after_layer = next(model.fc.parameters()).clone().detach()

before_layer[:, 0:50]

after_layer[:, 0:50]

(before_layer == after_layer).sum()

# Ok le réseau apprend bien !


