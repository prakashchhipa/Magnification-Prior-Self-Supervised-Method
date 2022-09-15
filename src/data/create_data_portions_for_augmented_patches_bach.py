'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import cv2 
import os, random, shutil, csv, copy 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


ref = '/home/datasets/BACH/folds_data/'

folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
parts_in_fold = ['train']
target_parts_in_fold = ['train_20', 'train_40', 'train_60', 'train_80']
category_list =['benign', 'insitu', 'invasive', 'normal']

output1 = '/home/datasets/BACH/folds_patches/'
output2 = '/home/datasets/BACH/folds_augmented_patches/'

for fold in folds:
    for data_part in parts_in_fold:
        for category in category_list:
            patch_suffix = os.path.join(fold, data_part, category)
            
            output = output2
            for patch in os.listdir(os.path.join(output, patch_suffix)):
                patch_path = os.path.join(output, patch_suffix, patch)
                
                for target_part in target_parts_in_fold:
                    for image_file in os.listdir(os.path.join(ref, fold, target_part, category)):
                        print(image_file.split(".")[0])
                        print(patch)
                        if image_file.split(".")[0] in patch:
                            Path(os.path.join(output, fold, target_part, category)).mkdir(parents=True, exist_ok=True)
                            dest = shutil.copy(patch_path, os.path.join(output, fold, target_part, category))
                