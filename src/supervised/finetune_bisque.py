'''Author- Prkaash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

from distutils.log import error
import errno
from multiprocessing.sharedctypes import Value
import numpy as np
import json
import argparse
import time
from tqdm import tqdm
import cv2
import logging
import sys, os, yaml, csv, pickle
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision.models as torch_models
from sklearn.metrics import f1_score
from supervised.apply.datasets import get_BisQue_data_loader, get_BisQue_testdata_loader
from supervised.apply.transform import resize_transform_bach_512,resize_transform_bach_224, resize_transform_bach_224_augmix,resize_transform_bach_512_augmix
from supervised.apply.augmentation_strategy import augmentation_03,augmentation_05, augmentation_08, augmentation_bach_03,augmentation_bach_08
from supervised.core.models import ResNet_Model, ResNetDilated_BACH
from supervised.core.train_util import Train_Util
import bc_config

import multiprocessing as mp
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
def create_config(config_exp):
    with open(config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def train_model(args_dict, fold):
    
    #1. Data settings
    data_path = args_dict["data_path"]
    train_data_portion = args_dict["train_data_portion"]
    #val_data_portion =  args_dict["val_data_portion"]
    test_data_portion =  args_dict["test_data_portion"]
    #barred = args_dict["barred"]

    #2. Model settings
    encoder = args_dict["encoder"]["name"]
    version = args_dict["encoder"]["version"]
    dropout = args_dict["encoder"]["fc_dropout"]
    pretrained_weights_type = args_dict["pretrained_encoder"]["pretrained_weights_type"]
    #3. Pretraining method settings
    pretraining_method_type = args_dict["pretrained_encoder"]["method_type"]
    pretraining_pair_sampling_method = args_dict["pretrained_encoder"]["variant"]
    pretraining_initial_weights = args_dict["pretrained_encoder"]["initial_weights"]
    pretraining_batch_size_list = args_dict["pretrained_encoder"]["batch_size_list"]
    #print(pretraining_batch_size_list)
    pretraining_epochs_list = args_dict["pretrained_encoder"]["epochs_list"]
    pretraining_checkpoint_base_path = args_dict["pretrained_encoder"]["checkpoint_base_path"]

    #4. Training (finetune) settings
    epochs = args_dict["epochs"]
    batch_size = args_dict["batch_size"]
    LR = args_dict["learning_rate"]["lr_only"]
    patience = args_dict["learning_rate"]["patience"]
    early_stopping_patience = args_dict["early_stopping_patience"]
    weight_decay = args_dict["weight_decay"]
    optimizer_choice = args_dict["optimizer"]
    augmentation_level = args_dict["augmentation_level"]
    momentum =args_dict["momentum"]
    input_image_size = args_dict["input_image_size"]
    linear_eval = args_dict["linear_eval"]
    
    #5. Computational infra settings
    gpu_no = (args_dict["computational_infra"]["fold_to_gpu_mapping"][fold])
    device = torch.device(f"cuda:{gpu_no}")
    workers = args_dict["computational_infra"]["workers"]

    #6. Logs and results settings
    tensorboard_base_path = args_dict["logs"]["tensorboard_base_path"]
    os.makedirs(tensorboard_base_path, exist_ok=True)
    result_base_path = args_dict["results"]["result_base_path"]
    os.makedirs(result_base_path, exist_ok=True)
    result_stats_path = args_dict["results"]["result_stats_path"]
    os.makedirs(result_stats_path, exist_ok=True)

    transform = None
    if 224 == input_image_size:
        transform = resize_transform_bach_224
    elif 512 == input_image_size:
        transform = resize_transform_bach_512
    else:
        raise error("input image size not supported as of now")
    
    augmentation_strategy = None
    if "bach_low" == augmentation_level:
        augmentation_strategy = augmentation_bach_03
    elif "bach_high" == augmentation_level:
        augmentation_strategy = augmentation_bach_08
    elif "low" == augmentation_level:
        augmentation_strategy = augmentation_03
    elif "moderate" == augmentation_level:
        augmentation_strategy = augmentation_05
    elif "high" == augmentation_level:
        augmentation_strategy = augmentation_08
    elif "none" == augmentation_level:
        augmentation_strategy = None
    else:
        raise error ("wrong input for augmentation level parameter")

    #train loader settings
    train_path = os.path.join(data_path, "folds_patches", fold, "train") #folds_augmented_patches, folds_patches
    allowed_data_list = []
    if train_data_portion in ["train"]:
        with open (os.path.join(data_path, "folds_metadata", fold, train_data_portion + ".ob"), 'rb') as fp:
            allowed_data_list = pickle.load(fp)
    else:
        raise error("Invalid train set name given")
    
    train_loader  = get_BisQue_data_loader(
        dataset_path= train_path,
        allowed_data_list = allowed_data_list,
        transform=transform,
        augmentation_strategy = augmentation_strategy,
        num_workers = workers
        )
    
    #test loader settings
    test_path = os.path.join(data_path, "folds_patches", fold, test_data_portion)
    test_allowed_data_list = []
    if test_data_portion in ["test"]:
        with open (os.path.join(data_path, "folds_metadata", fold, test_data_portion + ".ob"), 'rb') as fp:
            test_allowed_data_list = pickle.load(fp)
    else:
        raise error("Invalid test set name given")
    test_loader = get_BisQue_testdata_loader(
        dataset_path= test_path,
        allowed_data_list = test_allowed_data_list,
        transform = transform,
        num_workers = workers
        )

    #Experiment description
    DP = 0
    if "train" == train_data_portion:
        DP = 100
    else:
        raise ValueError("Invalid value for train data portion for BisQue dataset")
        
    if "imagenet" == pretraining_method_type:
        
        experiment_description = f"_{fold}_Bisque_FT_{DP}_{encoder}{version}_{pretraining_method_type}_{optimizer_choice}_{pretrained_weights_type}_{input_image_size}_"
        downstream_task_model = None
        if "resnet" == encoder:
            print(f"Start - loading weights for {fold}_{pretraining_method_type}_{encoder}{version}")
            downstream_task_model = ResNet_Model(version=int(version), pretrained=False)
            if pretrained_weights_type == "pytorch_imagenet":
                downstream_task_model = ResNet_Model(version=int(version), pretrained=True)
            elif pretrained_weights_type == "timm_imagenet":
                downstream_task_model.model.load_state_dict(torch.load("/home/models/timm_models/resnet50_a1_0-14fe96d1.pth"), strict=False)
            elif pretrained_weights_type == "timm_swsl_imagenet":
                downstream_task_model.model.load_state_dict(torch.load("/home/models/timm_models/semi_weakly_supervised_resnet50-16a12f1b.pth"), strict=False)
            elif pretrained_weights_type == "timm_ssl_resnet":
                downstream_task_model.model.load_state_dict(torch.load("/home/models/timm_models/semi_supervised_resnet50-08389792.pth"), strict=False)    
            num_ftrs=downstream_task_model.num_ftrs
            print(f"Stop - loading weights for {fold}_{pretraining_method_type}_{encoder}{version}")
            downstream_task_model.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 2))
        elif "dilated_resnet" == encoder:
                if 50 == int(version) and pretrained_weights_type == "imagenet":
                    downstream_task_model = ResNetDilated_BACH(version=int(version), num_classes=2)
                else:
                    error_message = f"{encoder} {pretrained_weights_type} {version} not supported"
                    raise error (error_message)

        
        downstream_task_model = downstream_task_model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        optimizer, scheduler = None, None
        if "sgd" == optimizer_choice:
            optimizer = torch.optim.SGD(downstream_task_model.parameters(), lr=LR, nesterov= True, momentum= momentum, dampening=0.0, weight_decay=weight_decay)
            scheduler = None
        elif "adam" == optimizer_choice:
            optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)    
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
        elif "adamw" == optimizer_choice:
            optimizer = torch.optim.AdamW(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)    
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
        else:
            raise error("input optimizer choice is wrong or not implemented")
        
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_base_path, experiment_description))
        train_util = Train_Util(
            experiment_description = experiment_description, 
            epochs = epochs, 
            model=downstream_task_model, 
            device=device, 
            train_loader=train_loader, 
            val_loader=test_loader,
            test_loader=test_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            batch_size=batch_size,
            scheduler=scheduler, 
            num_classes= len(bc_config.bach_label_list), 
            writer=writer, 
            early_stopping_patience = early_stopping_patience, 
            batch_balancing=False,
            threshold=None, 
            result_folder=result_base_path,
            linear_eval= linear_eval
            )
        results_dict = train_util.train_and_evaluate_bach()
        filepath = os.path.join(result_stats_path, f"{encoder}{version}_BACH_FT_{DP}_{pretraining_method_type}.csv")
        file_exists = os.path.isfile(filepath)
        with open(filepath, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['test_name','image_level_accuracy', 'image_level_weighted_f1', 'image_level_classwise_precision_0', 'image_level_classwise_precision_1','image_level_classwise_precision_2','image_level_classwise_precision_3','image_level_classwise_recall_0','image_level_classwise_recall_1', 'image_level_classwise_recall_2','image_level_classwise_recall_3','image_level_classwise_f1_0','image_level_classwise_f1_1', 'image_level_classwise_f1_2', 'image_level_classwise_f1_3', 'patch_level_accuracy', 'patch_level_weighted_f1', 'patch_level_classwise_precision_0', 'patch_level_classwise_precision_1','patch_level_classwise_precision_2','patch_level_classwise_precision_3','patch_level_classwise_recall_0','patch_level_classwise_recall_1', 'patch_level_classwise_recall_2','patch_level_classwise_recall_3','patch_level_classwise_f1_0','patch_level_classwise_f1_1', 'patch_level_classwise_f1_2', 'patch_level_classwise_f1_3'])
            for key in list (results_dict.keys()):
                best_acc_image_level, best_f1_image_level, best_classwise_precision_image_level, best_classwise_recall_image_level, best_classwise_f1_image_level, best_acc, best_f1, best_classwise_precision, best_classwise_recall, best_classwise_f1 = results_dict[key]
                writer.writerow([f"{fold}_{encoder}{version}_BISQUE_FT_{DP}_{pretraining_method_type}_aug_{augmentation_level}__ESC{early_stopping_patience}_INP{input_image_size}_LR{LR}_WD{weight_decay}_drop{dropout}_optimizer{optimizer_choice}_{key}", best_acc_image_level, best_f1_image_level, best_classwise_precision_image_level[0], best_classwise_precision_image_level[1],best_classwise_precision_image_level[2],best_classwise_precision_image_level[3],best_classwise_recall_image_level[0], best_classwise_recall_image_level[1],best_classwise_recall_image_level[2], best_classwise_recall_image_level[3],best_classwise_f1_image_level[0],best_classwise_f1_image_level[1], best_classwise_f1_image_level[2], best_classwise_f1_image_level[3], best_acc, best_f1, best_classwise_precision[0], best_classwise_precision[1],best_classwise_precision[2],best_classwise_precision[3],best_classwise_recall[0], best_classwise_recall[1],best_classwise_recall[2], best_classwise_recall[3],best_classwise_f1[0],best_classwise_f1[1], best_classwise_f1[2], best_classwise_f1[3]])
    else:
        for pretrained_encoder_dir in os.listdir(pretraining_checkpoint_base_path):
            
            if(pretraining_method_type in pretrained_encoder_dir) and (pretraining_pair_sampling_method in pretrained_encoder_dir) and (pretraining_initial_weights in pretrained_encoder_dir) and (encoder in pretrained_encoder_dir and str(version) in pretrained_encoder_dir):
                for pretrained_batch_size in os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir)):
                    if os.path.isdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir,pretrained_batch_size)):# and pretrained_batch_size in list(pretraining_batch_size_list):
                        _pretrained_batch_size = int(pretrained_batch_size)
                        if _pretrained_batch_size in list(pretraining_batch_size_list):
                            for pretrained_epoch in list(os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, str(pretrained_batch_size)))):
                                _pretrained_epoch = int(pretrained_epoch)
                                if _pretrained_epoch in list(pretraining_epochs_list):
                                    #Experiment description
                                    experiment_description = f"_{fold}_BISQUE_FT_{DP}_{encoder}{version}_{pretraining_method_type}_{pretraining_pair_sampling_method}_BS{pretrained_batch_size}_epoch{pretrained_epoch}_{pretraining_initial_weights}__LE{linear_eval}_"
                                    print ('experiment_description ', experiment_description)
                                    downstream_task_model = None
                                    if "resnet" == encoder:
                                        downstream_task_model = ResNet_Model(version=int(version), pretrained=False)
                                        num_ftrs=downstream_task_model.num_ftrs
                                        print(f"Start - loading weights for {fold}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{encoder}{version}_{pretraining_initial_weights}_{pretrained_batch_size}_{pretrained_epoch}")
                                        lst = os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch))
                                        lst.sort()
                                        model_path = os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch, lst[0])
                                        print ("model path - ", model_path)
                                        print ("GPU to be used - ", device)
                                        downstream_task_model.model.load_state_dict(torch.load(model_path), strict = False)
                                        print(f"Stop - loading weights for {fold}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{encoder}{version}_{pretraining_initial_weights}_{pretrained_batch_size}_{pretrained_epoch}")
                                        downstream_task_model.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 2))
                                    elif "dilated_resnet" == encoder:
                                        if 50 == int(version):
                                            downstream_task_model = ResNetDilated_BACH(version=int(version), num_classes=2)
                                            print(f"Start - loading weights for {fold}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{encoder}{version}_{pretraining_initial_weights}_{pretrained_batch_size}_{pretrained_epoch}")
                                            lst = os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch))
                                            lst.sort()
                                            model_path = os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch, lst[0])
                                            print ("model path - ", model_path)
                                            print ("GPU to be used - ", device)
                                            downstream_task_model.model.load_state_dict(torch.load(model_path), strict = False)
                                            print(f"Stop - loading weights for {fold}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{encoder}{version}_{pretraining_initial_weights}_{pretrained_batch_size}_{pretrained_epoch}")
                                    else:
                                        error_message = f"{encoder} {pretrained_weights_type} {version} not supported"
                                        raise error (error_message)

                                    downstream_task_model = downstream_task_model.to(device)
                                    criterion = nn.CrossEntropyLoss()
                                    
                                    optimizer, scheduler = None, None
                                    if "sgd" == optimizer_choice:
                                        optimizer = torch.optim.SGD(downstream_task_model.parameters(), lr=LR, nesterov=True, momentum=momentum, weight_decay=weight_decay)
                                        scheduler = None
                                    elif "adam" == optimizer_choice:
                                        optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)    
                                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
                                    elif "adamw" == optimizer_choice:
                                        optimizer = torch.optim.AdamW(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)    
                                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
                                    else:
                                        raise error("input optimzer choice is wrong or not implemented")
                                    
                                    writer = SummaryWriter(log_dir=os.path.join(tensorboard_base_path, experiment_description))
                                    train_util = Train_Util(
                                        experiment_description = experiment_description, 
                                        epochs = epochs, 
                                        model=downstream_task_model, 
                                        device=device, 
                                        train_loader=train_loader, 
                                        val_loader=test_loader,
                                        test_loader=test_loader, 
                                        optimizer=optimizer, 
                                        criterion=criterion, 
                                        batch_size=batch_size,
                                        scheduler=scheduler, 
                                        num_classes= 2,
                                        writer=writer, 
                                        early_stopping_patience = early_stopping_patience, 
                                        batch_balancing=False,
                                        threshold=None, 
                                        result_folder=result_base_path,
                                        linear_eval= linear_eval
                                        )
                                    results_dict = train_util.train_and_evaluate_bach()
                                    filepath = os.path.join(result_stats_path, f"{encoder}{version}_BISQUE_FT_{DP}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{pretrained_batch_size}_{pretrained_epoch}.csv")
                                    file_exists = os.path.isfile(filepath)
                                    with open(filepath, 'a') as f:
                                        writer = csv.writer(f)
                                        if not file_exists:
                                            writer.writerow(['test_name','image_level_accuracy', 'image_level_weighted_f1', 'image_level_classwise_precision_0', 'image_level_classwise_precision_1','image_level_classwise_precision_2','image_level_classwise_precision_3','image_level_classwise_recall_0','image_level_classwise_recall_1', 'image_level_classwise_recall_2','image_level_classwise_recall_3','image_level_classwise_f1_0','image_level_classwise_f1_1', 'image_level_classwise_f1_2', 'image_level_classwise_f1_3', 'patch_level_accuracy', 'patch_level_weighted_f1', 'patch_level_classwise_precision_0', 'patch_level_classwise_precision_1','patch_level_classwise_precision_2','patch_level_classwise_precision_3','patch_level_classwise_recall_0','patch_level_classwise_recall_1', 'patch_level_classwise_recall_2','patch_level_classwise_recall_3','patch_level_classwise_f1_0','patch_level_classwise_f1_1', 'patch_level_classwise_f1_2', 'patch_level_classwise_f1_3'])
                                        for key in list (results_dict.keys()):
                                            best_acc_image_level, best_f1_image_level, best_classwise_precision_image_level, best_classwise_recall_image_level, best_classwise_f1_image_level, best_acc, best_f1, best_classwise_precision, best_classwise_recall, best_classwise_f1 = results_dict[key]
                                            writer.writerow([f"{fold}_{encoder}{version}_BISQUE_FT_{DP}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{pretrained_batch_size}_{pretrained_epoch}_aug_{augmentation_level}__ESC{early_stopping_patience}_INP{input_image_size}_LR{LR}_WD{weight_decay}_drop{dropout}_optimizer{optimizer_choice}_{key}", best_acc_image_level, best_f1_image_level, best_classwise_precision_image_level[0], best_classwise_precision_image_level[1],best_classwise_precision_image_level[2],best_classwise_precision_image_level[3],best_classwise_recall_image_level[0], best_classwise_recall_image_level[1],best_classwise_recall_image_level[2], best_classwise_recall_image_level[3],best_classwise_f1_image_level[0],best_classwise_f1_image_level[1], best_classwise_f1_image_level[2], best_classwise_f1_image_level[3], best_acc, best_f1, best_classwise_precision[0], best_classwise_precision[1],best_classwise_precision[2],best_classwise_precision[3],best_classwise_recall[0], best_classwise_recall[1],best_classwise_recall[2], best_classwise_recall[3],best_classwise_f1[0],best_classwise_f1[1], best_classwise_f1[2], best_classwise_f1[3]])
                                    
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetuning  | Linear-evaluation on BisQue Breast Cancer Cell Dataset')
    parser.add_argument('--config', help='Config file for the experiment')
    args = parser.parse_args()
    args_dict = create_config(args.config)

    #mp.set_start_method('spawn')
    for fold in list(args_dict["computational_infra"]["fold_to_gpu_mapping"].keys()):
        process = mp.Process(target=train_model, args=(args_dict, fold))
        process.start()
