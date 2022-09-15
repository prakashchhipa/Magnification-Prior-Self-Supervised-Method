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
from empatches import EMPatches


root = '/home/datasets/BACH/folds_data/'

folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
parts_in_fold = ['train_05', 'train_10', 'train_20', 'train_40', 'train_60', 'train_80', 'train', 'test', 'val']
category_list =['benign', 'insitu', 'invasive', 'normal']

output = '/home/prachh/datasets/BACH/folds_patches/'


for fold in folds:
    for data_part in parts_in_fold:
        for category in category_list:
            path_suffix = os.path.join(fold, data_part, category)
            # list and access large image from root
            for large_image in os.listdir(os.path.join(root, path_suffix)):
                emp = EMPatches()
                img_patches, indices = emp.extract_patches(np.asarray(Image.open(os.path.join(root, path_suffix, large_image))), patchsize=512, overlap=0.5)
                print(f'{os.path.join(root, path_suffix, large_image)} patches count - ', len(img_patches))
                for idx in range(0, len(img_patches)):
                    name_splits = large_image.split(".")
                    patch_name = f"{name_splits[0]}_{idx}.{name_splits[1]}"
                    Path(os.path.join(output, path_suffix)).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(os.path.join(output, path_suffix, patch_name), img_patches[idx])
                
            
        
    