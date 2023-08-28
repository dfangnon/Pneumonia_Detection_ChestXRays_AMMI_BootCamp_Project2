#Imports
import os
import sys
import glob
import torch
import torchvision

import numpy    as np
import datetime as dt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot   as plt

from PIL               import Image
from torch.utils.data  import Dataset
from torch.autograd    import Variable
from torch.optim       import lr_scheduler

from torch.utils.data  import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision       import transforms, datasets, models
from os                import listdir, makedirs, getcwd, remove
from os.path           import isfile, join, abspath, exists, isdir, expanduser
import pandas as pd
from hypergrad import SGDHD, AdamHD
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import ttach as tta



def createModel():
  top_model = Sequential()
  top_model.add(ResNet152V2(include_top=False, weights="imagenet", input_shape=(224,224,3)))
  top_model.add(Reshape(7,7,2048), input_shape=(-1,7,7,2048))
  top_model.add(Flatten())
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation='sigmoid'))
  return top_model