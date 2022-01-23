'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import os
import torch
import cv2
import torch.nn as nn
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from skimage import io
import json
import cv2, os, random
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize
import albumentations as A
from skimage import io
import numpy as np

from supervised.apply import config

import bc_config

class BreakHis_Dataset(nn.Module):

    def __init__(self, train_path, transform = None, augmentation_strategy = None, pre_processing = [], image_type_list = []):

        # Standard setting for - dataset path, augmentation, trnformations, preprocessing, etc. 
        self.train_path = train_path
        self.transform = transform
        # No preprocessing as of noe
        self.pre_processing = pre_processing
        self.augmentation_strategy = augmentation_strategy
        self.image_type_list = image_type_list

        
        #key pairing for image examples
        self.image_dict_40x = {}
        self.image_dict_100x = {}
        self.image_dict_200x = {}
        self.image_dict_400x = {}

        #key pairing for labels
        self.label_binary_dict = {}
        self.label_multi_dict = {}

        for patient_dir_name in os.listdir(train_path):
            patient_uid = patient_dir_name.split('-')[1]
            binary_label = patient_dir_name.split('_')[1]
            multi_label = patient_dir_name.split('_')[2]

            
            
            #record keeping for 40X images
            path_40x = train_path + patient_dir_name + '/40X/'
            for image_name in os.listdir(path_40x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_40x[patient_uid+'_'+ image_seq] = path_40x + image_name
                #record keeping for binary label
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label
            
            #record keeping for 100X images
            path_100x = train_path + patient_dir_name + '/100X/'
            for image_name in os.listdir(path_100x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_100x[patient_uid+'_'+ image_seq] = path_100x + image_name
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label

            #record keeping for 200X images
            path_200x = train_path + patient_dir_name + '/200X/'
            for image_name in os.listdir(path_200x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_200x[patient_uid+'_'+ image_seq] = path_200x + image_name
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label

            #record keeping for 400X images
            path_400x = train_path + patient_dir_name + '/400X/'
            for image_name in os.listdir(path_400x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_400x[patient_uid+'_'+ image_seq] = path_400x + image_name
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label

        
        self.list_40X = list(self.image_dict_40x.keys())
        self.list_100X = list(self.image_dict_100x.keys())
        self.list_200X = list(self.image_dict_200x.keys())
        self.list_400X = list(self.image_dict_400x.keys())
        self.dict_list = {}
        self.dict_magnification_list = {}
        self.dict_magnification_list[bc_config.X40] = self.list_40X
        self.dict_magnification_list[bc_config.X100] = self.list_100X
        self.dict_magnification_list[bc_config.X200] = self.list_200X
        self.dict_magnification_list[bc_config.X400] = self.list_400X
        
        img_list = self.dict_magnification_list[self.image_type_list[0]]
        for magnification_level in self.image_type_list[1: len(self.image_type_list) - 1]:
            img_list = img_list & self.dict_magnification_list[magnification_level]
        self.image_list = img_list
        
        
                        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        item_dict = {}
        magnification = None
        patient_id = None
        if bc_config.X400 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X400] = PIL.Image.open(self.image_dict_400x[self.image_list[index]])
            magnification = bc_config.X400
        if bc_config.X200 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X200] = PIL.Image.open(self.image_dict_200x[self.image_list[index]])
            magnification = bc_config.X200
        if bc_config.X100 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X100] = PIL.Image.open(self.image_dict_100x[self.image_list[index]])
            magnification = bc_config.X100
        if bc_config.X40 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X40] = PIL.Image.open(self.image_dict_40x[self.image_list[index]])
            magnification = bc_config.X40
                       
        #Uniform augmentation
        state = torch.get_rng_state()
        if None != self.augmentation_strategy:
            for mg_level in list(item_dict.keys()):
                torch.set_rng_state(state)
                item_dict[mg_level] = self.augmentation_strategy(image = np.array(item_dict[mg_level]))

        if None != self.transform:
            for mg_level in list(item_dict.keys()):
                if None == self.augmentation_strategy:
                    if 0 == len(self.pre_processing):
                        item_dict[mg_level] = self.transform(np.array(item_dict[mg_level]))
                    else:
                        item_dict[mg_level] = self.transform(item_dict[mg_level])
                else:
                    item_dict[mg_level] = self.transform(item_dict[mg_level]['image'])
        
        return patient_id, magnification, item_dict, bc_config.binary_label_dict[self.label_binary_dict[self.image_list[index]]], bc_config.multi_label_dict[self.label_multi_dict[self.image_list[index]]]

def get_BreakHis_data_loader(dataset_path, transform = None, augmentation_strategy = None, pre_processing = None, image_type_list=[]):

    dataset = BreakHis_Dataset(train_path = dataset_path, transform = transform, augmentation_strategy = augmentation_strategy, pre_processing = pre_processing, image_type_list = image_type_list)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    return loader

def get_BreakHis_testdata_loader(dataset_path, transform = None, pre_processing = None, image_type_list=[]):

    dataset = BreakHis_Dataset(dataset_path, transform = transform, augmentation_strategy = None, pre_processing=pre_processing, image_type_list = image_type_list)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)
    return loader