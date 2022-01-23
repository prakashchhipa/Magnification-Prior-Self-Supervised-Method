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

from sklearn.metrics import f1_score,matthews_corrcoef

from supervised.core.models import EfficientNet_Model


from supervised.apply.datasets import get_BreakHis_data_loader, get_BreakHis_testdata_loader 
from supervised.apply.transform import train_transform, resize_transform
from supervised.apply.augmentation_strategy import ft_augmentation
from supervised.core.models import ResNet
from self_supervised.core.models import ResNet_MLP
from supervised.core.train_util import Train_Util
import bc_config


def get_metrics_from_confusion_matrix(confusion_matrix_epoch):

        #classwise precision
        epoch_classwise_precision_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu())/np.array(confusion_matrix_epoch.cpu()).sum(axis=0)
        epoch_classwise_precision_manual_cpu = np.nan_to_num(epoch_classwise_precision_manual_cpu, nan=0, neginf=0, posinf=0)
        #classwise recall
        epoch_classwise_recall_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu())/np.array(confusion_matrix_epoch.cpu()).sum(axis=1)
        epoch_classwise_recall_manual_cpu = np.nan_to_num(epoch_classwise_recall_manual_cpu, nan=0, neginf=0, posinf=0)
        #classwise f1
        epoch_classwise_f1_manual_cpu = (2*(epoch_classwise_precision_manual_cpu*epoch_classwise_recall_manual_cpu))/(epoch_classwise_precision_manual_cpu + epoch_classwise_recall_manual_cpu)
        epoch_classwise_f1_manual_cpu = np.nan_to_num(epoch_classwise_f1_manual_cpu, nan=0, neginf=0, posinf=0)
        #weighted average F1
        epoch_avg_f1_manual = np.sum(epoch_classwise_f1_manual_cpu*np.array(confusion_matrix_epoch.cpu()).sum(axis=1))/np.array(confusion_matrix_epoch.cpu()).sum(axis=1).sum()
        #accuracy
        epoch_acc_manual = 100*np.sum(np.array(confusion_matrix_epoch.diag().cpu()))/np.sum(np.array(confusion_matrix_epoch.cpu()))

        return epoch_avg_f1_manual, epoch_acc_manual, epoch_classwise_precision_manual_cpu, epoch_classwise_recall_manual_cpu, epoch_classwise_f1_manual_cpu


def test(model, test_loader, device, threshold):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        model.eval()
        
        with torch.no_grad():
            for item_dict, binary_label, multi_label in tqdm(test_loader):
                view = item_dict[magnification]
                view = view.cuda(device, non_blocking=True)                
                
                
                target = binary_label.to(device)

                outputs = model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)
               
                predicted = (outputs > threshold).int()

                predicted = predicted.to(device)

                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                    
                
        
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = get_metrics_from_confusion_matrix(confusion_matrix_val)
        

        print('MPCS pretrained fine-tuned on validation set: ')
        print('Testset classwise precision', classwise_precision)
        print('Testset classwise recall', classwise_recall)
        print('Testset classwise f1', classwise_f1)

        
        print('Testset Weighted F1',weighted_f1)
        print('Testset Accuracy', accuracy)

        return weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1

def test_model():
    
    threshold = ###
    fold_root = ###
    device = ###
    model_path = ###
    data_apth = ###
    magnification = ### 40x        
    
    
    test_loader = get_BreakHis_testdata_loader(data_path, transform = resize_transform,pre_processing=[], image_type_list= [magnification])
    
    model = EfficientNet_Model(pretrained=False)
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    

    test(model=model, test_loader=test_loader, device=device, threshold=threshold, magnification = magnification)

    





if __name__ == "__main__":

    test_model()
