# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from config import *

from torchvision.datasets import ImageFolder
from torchvision import transforms

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# +
trans = transforms.Compose([
    transforms.Resize(size=(250, 250)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
dataset = ImageFolder(DATA_PATH, transform=trans)

idx_to_class = {}
for key, val in dataset.class_to_idx.items():
    idx_to_class[val] = key
# -

screenshots = []
photos = []
for img, tgt in subset:
    if tgt == 0:
        photos.append(img)
    else:
        screenshots.append(img)
photos = torch.vstack(photos)
screenshots = torch.vstack(screenshots)

# - Récupérer l’onde (~50Hz)
# - Seuillage + compte
# - Feature engineering :
#     - Analyse de fourier (spectre) sur un échantillon
#     - Feature extraction -> principales fréquences, bruit, etc..
#     
#
# "TensorFlow is great, specially when you don’t use it"

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
        img = img.squeeze()
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        ind += 1

# +
fig, axs = plt.subplots(row_nb, col_nb, figsize=(row_nb*10, col_nb*10))

ind = 0
for i in range(row_nb):
    for j in range(col_nb):
        img, tgt = subset[ind]
        freq_img = torch.abs((torch.fft.fft2(img).squeeze()))
        freq_img = (freq_img >= 15.0) * 15.0 + (freq_img < 15.0) * freq_img
        frequences = torch.fft.fftfreq(img.numel())
        axs[i, j].imshow(freq_img)
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        ind += 1
# -












