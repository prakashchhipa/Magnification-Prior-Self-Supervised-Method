'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import numpy as np
import json
import argparse
import time
from tqdm import tqdm
import cv2
import logging
import sys, os

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from sklearn.metrics import f1_score



from efficientnet_pytorch.utils import MemoryEfficientSwish 

from supervised.apply.datasets import get_BreakHis_data_loader,get_BreakHis_testdata_loader 
from supervised.apply.transform import train_transform, resize_transform
from supervised.apply.augmentation_strategy import ft_augmentation,ft_exp_augmentation
from supervised.core.models import ResNet, EfficientNet_Model
from self_supervised.core.models import ResNet_MLP, EfficientNet_MLP
from supervised.core.train_util import Train_Util
import bc_config

def finetune_Effnet_b2(train_data_fold, test_data_fold, magnification, LR, epoch, description):
    
    LR = LR  #0.00002
    fold_root = data_fold_path
    experiment_description = description
    epoch = epoch
    model_path = mpcs_pretrained_model_path
    
    weight_decay = 5e-3        
    threshold =0.2 #0.5
    device = bc_config.gpu0

    train_loader  = get_BreakHis_data_loader(train_data_fold, transform=train_transform, augmentation_strategy=ft_exp_augmentation, pre_processing=[], image_type_list=[magnification])
    test_loader = get_BreakHis_testdata_loader(test_data_fold, transform = resize_transform, pre_processing=[], image_type_list= [magnification])
    
    #1. Initialize downstream model architecture
    downstream_task_model = EfficientNet_Model(pretrained=True)
    downstream_task_model = downstream_task_model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=10, min_lr= 5e-3)


    writer = SummaryWriter()
    train_util = Train_Util(experiment_description = experiment_description, epochs = epoch, model=downstream_task_model, device=device, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, criterion=criterion, batch_size=bc_config.batch_size,scheduler=scheduler, num_classes= len(bc_config.binary_label_list), writer=writer, threshold=threshold)
    train_util.train_and_evaluate()




if __name__ == "__main__":
    print('BreakHis Breast Cancer Dataset - Finetuning for pretrained network')
    
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--train_data_fold', type=str, required=True, help='The path for fold of trainset - chose data amount based on evaluation'
    )
    parser.add_argument(
        '--test_data_fold', type=str, required=True, help='The path for fold of testset'
    )
    parser.add_argument(
        '--magnification', type=str, required=True, help='Choose magnification for model fine-tuning, Example - 40x'
    )
    parser.add_argument(
        '--LR', type=float, required=False, default=0.00002, help='Learning Rate'
    )
    parser.add_argument(
        '--epoch', type=int, required=False, default=200, help=' No. of epochs'
    )
    parser.add_argument(
        '--description', type=str, required=False, help=' provide experiment description'
    )
    args = parser.parse_args()
    finetune_Effnet_b2(args.train_data_fold, args.test_data_fold, args.magnification, args.LR, args.epoch, args.description)
