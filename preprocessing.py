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

# +
row_nb = 5
col_nb = 5

indices = torch.randint(low=0, high=len(dataset), size=(row_nb*col_nb,))
subset = Subset(dataset, indices)

screenshots = []
photos = []
for img, tgt in subset:
    if tgt == 0:
        photos.append(img)
    else:
        screenshots.append(img)
photos = torch.vstack(photos)
screenshots = torch.vstack(screenshots)
# -

# - Récupérer l’onde (~50Hz)
# - Seuillage + compte
# - Feature engineering :
#     - Analyse de fourier (spectre) sur un échantillon
#     - Feature extraction -> principales fréquences, bruit, etc..
#     
#
# "TensorFlow is great, specially when you don’t use it"

# +
# %matplotlib inline
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
# -

# ## Analyse de la Magnitude

# +
# %matplotlib inline
fig, axs = plt.subplots(row_nb, col_nb, figsize=(row_nb*10, col_nb*10))

ind = 0
for i in range(row_nb):
    for j in range(col_nb):
        img, tgt = subset[ind]
        fft_img = torch.fft.fft2(img).squeeze()
        freq_img = torch.abs((fft_img))
        freq_img = torch.log10(freq_img)
        freq_x = torch.fft.fftshift(torch.fft.fftfreq(img.shape[2])).numpy()
        freq_y = torch.fft.fftshift(torch.fft.fftfreq(img.shape[1])).numpy()
        x_range = np.hstack((np.arange(0, img.shape[2], 50), np.array([img.shape[2] - 1]), np.array([img.shape[2] // 2])))
        y_range = np.hstack((np.arange(0, img.shape[1], 50), np.array([img.shape[1] - 1]), np.array([img.shape[2] // 2])))
        axs[i, j].set_xticks(x_range)
        axs[i, j].set_xticklabels(freq_x[x_range])
        axs[i, j].set_yticks(y_range)
        axs[i, j].set_yticklabels(freq_y[y_range])
        imshow = axs[i, j].imshow(freq_img)
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        fig.colorbar(imshow, ax=axs[i, j])
        ind += 1
# -
# ## Analyse de la phase

# +
# %matplotlib inline
fig, axs = plt.subplots(row_nb, col_nb, figsize=(row_nb*10, col_nb*10))

ind = 0
for i in range(row_nb):
    for j in range(col_nb):
        img, tgt = subset[ind]
        fft_img = torch.fft.fft2(img).squeeze()
        freq_img = torch.angle(fft_img)
        freq_x = torch.fft.fftshift(torch.fft.fftfreq(img.shape[2])).numpy()
        freq_y = torch.fft.fftshift(torch.fft.fftfreq(img.shape[1])).numpy()
        x_range = np.hstack((np.arange(0, img.shape[2], 50), np.array([img.shape[2] - 1]), np.array([img.shape[2] // 2])))
        y_range = np.hstack((np.arange(0, img.shape[1], 50), np.array([img.shape[1] - 1]), np.array([img.shape[2] // 2])))
        axs[i, j].set_xticks(x_range)
        axs[i, j].set_xticklabels(freq_x[x_range])
        axs[i, j].set_yticks(y_range)
        axs[i, j].set_yticklabels(freq_y[y_range])
        imshow = axs[i, j].imshow(freq_img)
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        fig.colorbar(imshow, ax=axs[i, j])
        ind += 1
# -

# ## Reconstruction selon phase

# +
# %matplotlib inline
fig, axs = plt.subplots(row_nb, col_nb, figsize=(row_nb*10, col_nb*10))

ind = 0
for i in range(row_nb):
    for j in range(col_nb):
        img, tgt = subset[ind]
        fft_img = torch.fft.fft2(img).squeeze()
        freq_img = torch.angle(fft_img)
        img_phase = torch.fft.ifft2(freq_img)
        imshow = axs[i, j].imshow(torch.real(img_phase))
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        ind += 1
# -

# ## Reconstruction selon magnitude

# +
# %matplotlib inline
fig, axs = plt.subplots(row_nb, col_nb, figsize=(row_nb*10, col_nb*10))

ind = 0
for i in range(row_nb):
    for j in range(col_nb):
        img, tgt = subset[ind]
        fft_img = torch.fft.fft2(img).squeeze()
        freq_img = torch.abs(fft_img)
        img_magnitude = torch.fft.ifft2(freq_img)
        imshow = axs[i, j].imshow(torch.real(img_magnitude))
        axs[i, j].set_title(idx_to_class[tgt], fontsize=40)
        ind += 1
# -






