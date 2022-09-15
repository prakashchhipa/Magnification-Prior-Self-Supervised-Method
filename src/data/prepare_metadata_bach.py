'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
import os, random, shutil, csv, copy 
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold,StratifiedKFold, train_test_split
from collections import Counter
from pathlib import Path
import pickle


root = '/home/datasets/BACH/folds_data/'


folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
parts_in_fold = ['train_20']
category_list =['benign', 'insitu', 'invasive', 'normal']

output = '/home/prachh/datasets/BACH/folds_metadata/'

for fold in folds:
    for data_part in parts_in_fold:
        list_20, list_10,list_05 = [],[],[]
        with open (os.path.join(output, fold, data_part + ".ob"), 'rb') as fp:
            list_20 = pickle.load(fp)
            #bengin
            list_10.extend(list_20[0:6])
            list_05.extend(list_20[0:3])
            #is
            list_10.extend(list_20[12:18])
            list_05.extend(list_20[12:15])
            #iv
            list_10.extend(list_20[24:30])
            list_05.extend(list_20[24:27])
            #normal
            list_10.extend(list_20[36:42])
            list_05.extend(list_20[36:39])
            
            with open(os.path.join(output, fold, "train_10" + ".ob"), 'wb') as fp:
                pickle.dump(list_10, fp)
            with open(os.path.join(output, fold, "train_05" + ".ob"), 'wb') as fp:
                pickle.dump(list_05, fp)
            
            with open (os.path.join(output, fold, "train_10" + ".ob"), 'rb') as fp:
                list_10 = pickle.load(fp)
                print (f'counts for {os.path.join(output, fold, "train_10.ob")}', len(list_10), list_10)
            with open (os.path.join(output, fold, "train_05" + ".ob"), 'rb') as fp:
                list_05 = pickle.load(fp)
                print (f'counts for {os.path.join(output, fold, "train_10.ob")}', len(list_05), list_05)
        
            
        
    