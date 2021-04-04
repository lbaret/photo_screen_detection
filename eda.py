# # EDA (Exploratory Data Analysis)

from torchvision.datasets import ImageFolder
from config import *
from PIL import Image
import numpy as np

# # Data loading

dataloader = ImageFolder(DATA_PATH)

dataloader

img, target = next(iter(dataloader))

img.convert('')


