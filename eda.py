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

# **TODO** : Penser à faire de la data augmentation.



