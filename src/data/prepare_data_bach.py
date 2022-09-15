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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from collections import Counter
from pathlib import Path

###%% READ DATA AND LABELS

k_folds = 5
random_state = None #give random seed (20 ~ 100)

root = '/home/datasets/BACH/images_stain_normalized/'
fold_data_path = '/home/datasets/BACH/folds_data/'

category_list =['benign', 'insitu', 'invasive', 'normal']

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
        
        #val - 20%
        temp_image_list_train, temp_image_list_val, temp_category_list_train, temp_category_list_val = train_test_split(train_image_list, train_cat_list ,stratify= train_cat_list, test_size=0.25, random_state=random_state)
        test_image_list = [images_list[index] for index in test_ids]
        test_cat_list = [cat_list[index] for index in test_ids]
        
        
        train_path = os.path.join(fold_data_path, f"fold_{fold}", "train" , category)
        Path(train_path).mkdir(parents=True, exist_ok=True)
        
        train_05_path = os.path.join(fold_data_path, f"fold_{fold}", "train_05" , category)
        Path(train_05_path).mkdir(parents=True, exist_ok=True)
        train_10_path = os.path.join(fold_data_path, f"fold_{fold}", "train_10" , category)
        Path(train_10_path).mkdir(parents=True, exist_ok=True)
        
        train_20_path = os.path.join(fold_data_path, f"fold_{fold}", "train_20" , category)
        Path(train_20_path).mkdir(parents=True, exist_ok=True)
        train_40_path = os.path.join(fold_data_path, f"fold_{fold}", "train_40" , category)
        Path(train_40_path).mkdir(parents=True, exist_ok=True)
        train_60_path = os.path.join(fold_data_path, f"fold_{fold}", "train_60" , category)
        Path(train_60_path).mkdir(parents=True, exist_ok=True)
        train_80_path = os.path.join(fold_data_path, f"fold_{fold}", "train_80" , category)
        Path(train_80_path).mkdir(parents=True, exist_ok=True)
        
        
        val_path = os.path.join(fold_data_path, f"fold_{fold}", "val" , category)
        Path(val_path).mkdir(parents=True, exist_ok=True)
        test_path = os.path.join(fold_data_path, f"fold_{fold}", "test" , category)
        Path(test_path).mkdir(parents=True, exist_ok=True)
        
        for idx in range(0, len(temp_image_list_train)):
            dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_path)
            
        for idx in range(0, len(temp_image_list_val)):
            dest = shutil.copy(os.path.join(category_path,temp_image_list_val[idx]), val_path)
        
        for idx in range(0, len(test_image_list)):
            dest = shutil.copy(os.path.join(category_path,test_image_list[idx]), test_path)
        
        
        #train
        for idx in range(0, len(temp_image_list_train)):
            if idx < 3:
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_05_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_10_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_20_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_40_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_60_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_80_path)
            if idx < 6:
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_10_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_20_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_40_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_60_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_80_path)
            elif idx < 12:
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_20_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_40_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_60_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_80_path)
            elif idx < 24:
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_40_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_60_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_80_path)
            elif idx < 36:
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_60_path)
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_80_path)
            elif idx < 48:
                dest = shutil.copy(os.path.join(category_path,temp_image_list_train[idx]), train_80_path)