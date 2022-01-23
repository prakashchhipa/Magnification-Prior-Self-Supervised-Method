'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import os
import torch
import cv2
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset
import json
import cv2, os
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize
import albumentations as A
from skimage import io
import numpy as np
from random import randrange

from self_supervised.apply import config

import bc_config

class BreakHis_Dataset_SSL(nn.Module):

    #Default pair sampling - Ordered Pair
    def __init__(self, train_path, training_method=None, transform = None, target_transform = None, augmentation_strategy = None, pre_processing = [], image_pair = [], pair_sampling_strategy = bc_config.OP):
        
        self.train_path = train_path
        self.transform = transform
        self.target_transform = target_transform
        # no preprocessing as of now
        self.pre_processing = pre_processing
        self.pair_sampling_strategy = pair_sampling_strategy
        self.image_dict_40x = {}
        self.image_dict_100x = {}
        self.image_dict_200x = {}
        self.image_dict_400x = {}
        
        #preprocessing - not in use now
        
        
        for patient_dir_name in os.listdir(train_path):
            patient_uid = patient_dir_name.split('-')[1]
            
            #record keeping for 40X images
            path_40x = train_path + patient_dir_name + '/40X/'
            for image_name in os.listdir(path_40x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_40x[patient_uid+'_'+ image_seq] = path_40x + image_name
            
            #record keeping for 100X images
            path_100x = train_path + patient_dir_name + '/100X/'
            for image_name in os.listdir(path_100x):
                image_seq = image_name.split('.')[0].split('-')[4]
                if (patient_uid+'_'+ image_seq in list(self.image_dict_40x.keys())):
                    self.image_dict_100x[patient_uid+'_'+ image_seq] = path_100x + image_name

            #record keeping for 200X images
            path_200x = train_path + patient_dir_name + '/200X/'
            for image_name in os.listdir(path_200x):
                image_seq = image_name.split('.')[0].split('-')[4]
                if ((patient_uid+'_'+ image_seq in list(self.image_dict_40x.keys())) and (patient_uid+'_'+ image_seq in list(self.image_dict_100x.keys()))):
                    self.image_dict_200x[patient_uid+'_'+ image_seq] = path_200x + image_name

            #record keeping for 400X images
            path_400x = train_path + patient_dir_name + '/400X/'
            for image_name in os.listdir(path_400x):
                image_seq = image_name.split('.')[0].split('-')[4]
                if ((patient_uid+'_'+ image_seq in list(self.image_dict_40x.keys())) and (patient_uid+'_'+ image_seq in list(self.image_dict_100x.keys())) and (patient_uid+'_'+ image_seq in list(self.image_dict_200x.keys()))):
                    self.image_dict_400x[patient_uid+'_'+ image_seq] = path_400x + image_name


        #SSL specific
        self.augmentation_strategy_1 = augmentation_strategy
        self.training_method = training_method
        self.image_pair = image_pair
        
        self.list_40X = list(self.image_dict_40x.keys())
        self.list_100X = list(self.image_dict_100x.keys())
        self.list_200X = list(self.image_dict_200x.keys())
        self.list_400X = list(self.image_dict_400x.keys())
        temp = list(set(self.list_40X) & set(self.list_100X) & set(self.list_200X) & set(self.list_400X))
        self.image_list = temp
                
                        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        image1_path, image2_path = None,None
        
        #Ordered Pair
        if bc_config.OP == self.pair_sampling_strategy:
            randon_mgnification = randrange(4)
            if randon_mgnification == 0:
                image1_path = self.image_dict_40x[self.image_list[index]]
                image2_path = self.image_dict_100x[self.image_list[index]]
            elif randon_mgnification == 1:
                image1_path = self.image_dict_100x[self.image_list[index]]
                image2_path = self.image_dict_200x[self.image_list[index]]
            elif randon_mgnification == 2:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image2_path = self.image_dict_400x[self.image_list[index]]
            elif randon_mgnification == 3:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image2_path = self.image_dict_400x[self.image_list[index]]
        
        #Random Pair
        if bc_config.RP == self.pair_sampling_strategy:
            randon_mgnification_1 = randrange(4)
            randon_mgnification_2 = randrange(4)
            #same magnification - not allowed
            while randon_mgnification_1 == randon_mgnification_2:
                randon_mgnification_2 = randrange(4)
            
            if randon_mgnification_1 == 0:
                image1_path = self.image_dict_40x[self.image_list[index]]
            elif randon_mgnification_1 == 1:
                image1_path = self.image_dict_100x[self.image_list[index]]
            elif randon_mgnification_1 == 2:
                image1_path = self.image_dict_200x[self.image_list[index]]
            elif randon_mgnification_1 == 3:
                image1_path = self.image_dict_400x[self.image_list[index]]

            if randon_mgnification_2 == 0:
                image2_path = self.image_dict_40x[self.image_list[index]]
            elif randon_mgnification_2 == 1:
                image2_path = self.image_dict_100x[self.image_list[index]]
            elif randon_mgnification_2 == 2:
                image2_path = self.image_dict_200x[self.image_list[index]]
            elif randon_mgnification_2 == 3:
                image2_path = self.image_dict_400x[self.image_list[index]]

        # Fixed Pair
        if bc_config.FP == self.pair_sampling_strategy:
            image1_path = self.image_dict_200x[self.image_list[index]]
            image2_path = self.image_dict_400x[self.image_list[index]]
        
        image1  = PIL.Image.open(image1_path)
        image2  = PIL.Image.open(image2_path)
        
        transformed_view1, transformed_view2 = None, None
               
        if self.training_method == config.MPCS:
            state = torch.get_rng_state()
            transformed_view1 = self.augmentation_strategy_1(image = np.array(image1))
            torch.set_rng_state(state)
            transformed_view2 = self.augmentation_strategy_1(image = np.array(image2))

            if self.transform:
                transformed_view1 = self.transform(transformed_view1['image'])
                transformed_view2 = self.transform(transformed_view2['image'])

            return transformed_view1, transformed_view2


def get_BreakHis_trainset_loader(train_path, training_method=None, transform = None,target_transform = None, augmentation_strategy = None, pre_processing = None, image_pair=[], pair_sampling_strategy = None):
    # no addtional preprocessing as of now
    dataset = BreakHis_Dataset_SSL(train_path, training_method, transform, target_transform, augmentation_strategy, pre_processing, image_pair, pair_sampling_strategy)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    return train_loader