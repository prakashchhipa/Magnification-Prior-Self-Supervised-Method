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

random_state = None #give random seed (20 ~ 100)

root = '/home/datasets/BreaKHis_v1/histology_slides/breast'

#Description - Benign
benign_list = ['/benign/SOB/adenosis/','/benign/SOB/fibroadenoma/', '/benign/SOB/phyllodes_tumor/','/benign/SOB/tubular_adenoma/']
#Description - Malignant
malignant_list = ['/malignant/SOB/lobular_carcinoma/', '/malignant/SOB/papillary_carcinoma/', '/malignant/SOB/ductal_carcinoma/', '/malignant/SOB/mucinous_carcinoma/']
count =0
patient_list = []
abstract_category_list = []
concrete_category_list = []

#Access benign categories patients
for benign_type_dir in benign_list:
    p_dir_path = root + benign_type_dir
    for p_id in os.listdir(p_dir_path):
        patient_list.append(p_dir_path + p_id)
        count +=1

#Access malignant categories patients
for malignant_type_dir in malignant_list:
    p_dir_path = root + malignant_type_dir
    for p_id in os.listdir(p_dir_path):
        patient_list.append(p_dir_path + p_id)
        count +=1

#Random shuffle the list and extract labels
random.Random(random_state).shuffle(patient_list)

with open(root +'/data.csv', 'w') as f:
    writer = csv.writer(f)
    for patient_path in patient_list:
        main_class = patient_path.split('/')[-1].split('_')[1]
        abstract_category_list.append(main_class)
        sub_class = patient_path.split('/')[-1].split('_')[2]
        concrete_category_list.append(sub_class)
        print(patient_path, main_class, sub_class)
        writer.writerow([patient_path, main_class, sub_class])

print('patient count', count)


k_folds = 5
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
stat_dict = {}
stat_dict_test = {}
stat_dict_val = {}
data_splits = kfold.split(patient_list,concrete_category_list)

for fold, (train_ids, test_ids) in enumerate(data_splits):
    with open(root +f'/fold{fold}_stat.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['part', 'benign', 'malignant', 'DC', 'LC', 'MC', 'PC', 'PT', 'F', 'TA', 'A'])
        fold_path = '/home/datasets/BreaKHis_v1/histology_slides/breast/Fold_'+ str(fold) + '_' + str(k_folds)
        Path(fold_path).mkdir(parents=True, exist_ok=True)
        stat_dict['B'] = 0
        stat_dict['M'] = 0
        stat_dict['DC'] = 0
        stat_dict['LC'] = 0
        stat_dict['MC'] = 0
        stat_dict['PC'] = 0
        stat_dict['PT'] = 0
        stat_dict['F'] = 0
        stat_dict['TA'] = 0
        stat_dict['A'] = 0
        stat_dict_test = copy.deepcopy(stat_dict)
        stat_dict_val = copy.deepcopy(stat_dict)

        print(f'FOLD {fold}')
        print('--------------------------------')
        temp_abstract_category_list = [abstract_category_list[index] for index in train_ids]
        temp_concrete_category_list = [concrete_category_list[index] for index in train_ids]
        temp_patient_list = [patient_list[index] for index in train_ids]

        #val - 20%
        temp_patient_list_train, temp_patient_list_val, temp_abstract_category_list_train, temp_abstract_category_list_val = train_test_split(temp_patient_list, temp_concrete_category_list ,stratify= temp_concrete_category_list, test_size=0.25, random_state=random_state)
        
        temp_abstract_category_list_test = [abstract_category_list[index] for index in test_ids]
        temp_patient_list_test = [patient_list[index] for index in test_ids]

        #train data move
        fold_path_train = fold_path + '/train_60/'
        Path(fold_path_train).mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list_train:
            main_class = patient.split('/')[-1].split('_')[1]
            sub_class = patient.split('/')[-1].split('_')[2]
            stat_dict[main_class] += 1
            stat_dict[sub_class] += 1
            dest = shutil.copytree(patient, fold_path_train + patient.split('/')[-1])
        
        #val data move
        fold_path_val = fold_path + '/val_20/'
        Path(fold_path_val).mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list_val:
            main_class = patient.split('/')[-1].split('_')[1]
            sub_class = patient.split('/')[-1].split('_')[2]
            stat_dict_val[main_class] += 1
            stat_dict_val[sub_class] += 1
            dest = shutil.copytree(patient, fold_path_val + patient.split('/')[-1])

        #test data move
        fold_path_test = fold_path + '/test_20/'
        Path(fold_path_test).mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list_test:
            main_class = patient.split('/')[-1].split('_')[1]
            sub_class = patient.split('/')[-1].split('_')[2]
            stat_dict_test[main_class] += 1
            stat_dict_test[sub_class] += 1
            dest = shutil.copytree(patient, fold_path_test+ patient.split('/')[-1])


        writer.writerow(['train_60', stat_dict['B'], stat_dict['M'], stat_dict['DC'], stat_dict['LC'], stat_dict['MC'], stat_dict['PC'], stat_dict['PT'], stat_dict['F'], stat_dict['TA'], stat_dict['A']])
        writer.writerow(['val_20', stat_dict_val['B'], stat_dict_val['M'], stat_dict_val['DC'], stat_dict_val['LC'], stat_dict_val['MC'], stat_dict_val['PC'], stat_dict_val['PT'], stat_dict_val['F'], stat_dict_val['TA'], stat_dict_val['A']])
        writer.writerow(['test_20', stat_dict_test['B'], stat_dict_test['M'], stat_dict_test['DC'], stat_dict_test['LC'], stat_dict_test['MC'], stat_dict_test['PC'], stat_dict_test['PT'], stat_dict_test['F'], stat_dict_test['TA'], stat_dict_test['A']])

        stat_dict['B'] = 0
        stat_dict['M'] = 0
        stat_dict['DC'] = 0
        stat_dict['LC'] = 0
        stat_dict['MC'] = 0
        stat_dict['PC'] = 0
        stat_dict['PT'] = 0
        stat_dict['F'] = 0
        stat_dict['TA'] = 0
        stat_dict['A'] = 0
        stat_dict_test = copy.deepcopy(stat_dict)
        stat_dict_val = copy.deepcopy(stat_dict)
        
        #val - 10%
        temp_patient_list_train, temp_patient_list_val, temp_abstract_category_list_train, temp_abstract_category_list_val = train_test_split(temp_patient_list, temp_concrete_category_list ,stratify= temp_concrete_category_list, test_size=0.125, random_state=random_state)
        
        temp_abstract_category_list_test = [abstract_category_list[index] for index in test_ids]
        temp_patient_list_test = [patient_list[index] for index in test_ids]

        #train data move
        fold_path_train = fold_path + '/train_70/'
        Path(fold_path_train).mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list_train:
            main_class = patient.split('/')[-1].split('_')[1]
            sub_class = patient.split('/')[-1].split('_')[2]
            stat_dict[main_class] += 1
            stat_dict[sub_class] += 1
            dest = shutil.copytree(patient, fold_path_train + patient.split('/')[-1])
        
        #val data move
        fold_path_val = fold_path + '/val_10/'
        Path(fold_path_val).mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list_val:
            main_class = patient.split('/')[-1].split('_')[1]
            sub_class = patient.split('/')[-1].split('_')[2]
            stat_dict_val[main_class] += 1
            stat_dict_val[sub_class] += 1
            dest = shutil.copytree(patient, fold_path_val + patient.split('/')[-1])


        train_len = len(temp_patient_list_train)
        val_len = len(temp_patient_list_val)
        test_len = len(temp_patient_list_test)
        print(f'Fold {fold} - count (train/val/test): {train_len}, {val_len}, {test_len}')
        writer.writerow(['train_70', stat_dict['B'], stat_dict['M'], stat_dict['DC'], stat_dict['LC'], stat_dict['MC'], stat_dict['PC'], stat_dict['PT'], stat_dict['F'], stat_dict['TA'], stat_dict['A']])
        writer.writerow(['val_10', stat_dict_val['B'], stat_dict_val['M'], stat_dict_val['DC'], stat_dict_val['LC'], stat_dict_val['MC'], stat_dict_val['PC'], stat_dict_val['PT'], stat_dict_val['F'], stat_dict_val['TA'], stat_dict_val['A']])