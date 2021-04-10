# -*- coding: utf-8 -*-
# # EDA (Exploratory Data Analysis)

# +
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import Subset, DataLoader

from config import *
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
# -

# # Data loading

trans = transforms.Compose([
    transforms.Resize(size=(1500, 1500)),
    transforms.ToTensor()
])
dataset = ImageFolder(DATA_PATH, transform=trans)

dataset

dataset.class_to_idx

# C’est quand même mieux d’avoir l’association label --> string

idx_to_class = {}
for key, val in dataset.class_to_idx.items():
    idx_to_class[val] = key

idx_to_class

# **On observe quelques images d’exemples**

# +
row_nb = 5
col_nb = 5

indices = torch.randint(low=0, high=len(dataset), size=(row_nb*col_nb,))
subset = Subset(dataset, indices)

fig, axs = plt.subplots(row_nb, col_nb, figsize=(row_nb*10, col_nb*10))

ind = 0
for i in range(row_nb):
    for j in range(col_nb):
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

# # Taille des images (max / min)
#
# Cela va nous permettre, pour la suite, de padder les images.

# +
maxH = 0
maxW = 0

minH = float('inf')
minW = float('inf')

# Tensor : (C, H, W)
for img, _ in dataset:
    if img.shape[1] > maxH:
        maxH = img.shape[1]
    if img.shape[2] > maxW:
        maxW = img.shape[2]
    if img.shape[1] < minH:
        minH = img.shape[1]
    if img.shape[2] < minW:
        minW = img.shape[2]
# -

maxH, maxW, minH, minW

# - Hauteur max : 3008
# - Hauteur min : 50
# - Largeur max : 3015
# - Largeur min : 192

# **TODO** : Penser à faire de la data augmentation.









