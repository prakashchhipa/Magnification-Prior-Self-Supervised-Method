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


root = '/home/datasets/BisQue/folds_data/'

folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
parts_in_fold = ['train', 'test']
category_list =['benign', 'malignant']

output = '/home/prachh/datasets/BisQue/folds_metadata/'

for fold in folds:
    for data_part in parts_in_fold:
        data_part_image_name_prefix_list = []   
        Path(os.path.join(output, fold)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output, fold, data_part + ".ob"), 'wb') as fp:
            for category in category_list:
                path_suffix = os.path.join(fold, data_part, category)
                # list and access large image from root
                for large_image in os.listdir(os.path.join(root, path_suffix)):
                    data_part_image_name_prefix_list.append(large_image.split(".")[0])
            pickle.dump(data_part_image_name_prefix_list, fp)
        
        #verifying the input entries
        with open (os.path.join(output, fold, data_part + ".ob"), 'rb') as fp:
            list_1 = pickle.load(fp)
            print (f'counts for {os.path.join(output, fold, data_part + ".ob")}', len(list_1), list_1)
            
        
    