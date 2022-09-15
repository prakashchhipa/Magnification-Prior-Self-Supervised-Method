'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import numpy as np
from PIL import Image
import PIL
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


root = '/home/datasets/BACH/folds_patches/'

folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
parts_in_fold = ['train', 'test', 'val' ,'train_20', 'train_40', 'train_60', 'train_80']
category_list =['benign', 'insitu', 'invasive', 'normal']

output = '/home/datasets/BACH/folds_augmented_patches/'

for fold in folds:
    for data_part in parts_in_fold:
        for category in category_list:
            path_suffix = os.path.join(fold, data_part, category)
            # list and access large image from root
            Path(os.path.join(output, path_suffix)).mkdir(parents=True, exist_ok=True)
            for patch in os.listdir(os.path.join(root, path_suffix)):
                name_splits = patch.split(".")
                im = Image.open(os.path.join(root, path_suffix, patch))
                
                #original image - also needs to be saved
                im.save(os.path.join(output, path_suffix, f"{name_splits[0]}.{name_splits[1]}"))
                
                #horizontal flip
                out_hf = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                out_hf.save(os.path.join(output, path_suffix, f"{name_splits[0]}_hf.{name_splits[1]}"))
                
                #vertical flip
                out_vf = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                out_vf.save(os.path.join(output, path_suffix, f"{name_splits[0]}_vf.{name_splits[1]}"))
                
                #90 degree
                out_90d = im.transpose(PIL.Image.ROTATE_90)
                out_90d.save(os.path.join(output, path_suffix, f"{name_splits[0]}_90d.{name_splits[1]}"))
                
                #180 degree
                out_180d = im.transpose(PIL.Image.ROTATE_180)
                out_180d.save(os.path.join(output, path_suffix, f"{name_splits[0]}_180d.{name_splits[1]}"))
                
                #270 degree
                out_270d = im.transpose(PIL.Image.ROTATE_270)
                out_270d.save(os.path.join(output, path_suffix, f"{name_splits[0]}_270d.{name_splits[1]}"))
                
                print(f"{os.path.join(root, path_suffix, patch)} patches augmented views created and saved")
    
            
            
        
    