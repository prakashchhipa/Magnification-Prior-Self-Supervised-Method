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

###%% READ DATA AND LABELS

k_folds = 5
random_state = None #give random seed (20 ~ 100)

root = '/home/datasets/BisQue/images/'
fold_data_path = '/home/datasets/BisQue/images/folds_data/'

category_list =['benign', 'malignant']

for category in category_list:
    category_path = root + category
    images_list = []
    cat_list = []
    for img in os.listdir(category_path):
        images_list.append(img)
        cat_list.append(category)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    data_splits = kfold.split(images_list,cat_list)
    
    for fold, (train_ids, test_ids) in enumerate(data_splits):
        print(f'FOLD {fold}')
        train_image_list = [images_list[index] for index in train_ids]
        train_cat_list = [cat_list[index] for index in train_ids]
        
        test_image_list = [images_list[index] for index in test_ids]
        test_cat_list = [cat_list[index] for index in test_ids]
        
        
        train_path = os.path.join(fold_data_path, f"fold_{fold}", "train" , category)
        Path(train_path).mkdir(parents=True, exist_ok=True)
        
        
        test_path = os.path.join(fold_data_path, f"fold_{fold}", "test" , category)
        Path(test_path).mkdir(parents=True, exist_ok=True)
        
        for idx in range(0, len(train_image_list)):
            dest = shutil.copy(os.path.join(category_path,train_image_list[idx]), train_path)
        
        for idx in range(0, len(test_image_list)):
            dest = shutil.copy(os.path.join(category_path,test_image_list[idx]), test_path)