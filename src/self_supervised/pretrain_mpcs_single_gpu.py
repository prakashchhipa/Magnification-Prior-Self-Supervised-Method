'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import argparse
import logging
import os, sys, yaml
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from self_supervised.core import ssl_loss, models, pretrain, utility, trainer_MPCS
sys.path.append(os.path.dirname(__file__))
from self_supervised.apply import datasets, config, transform, augmentation_strategy
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_WARNINGS"] = "FALSE"

import bc_config

import multiprocessing as mp


def create_config(config_exp):
    with open(config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    return config



def pretrain_model(args_dict, fold):
    
    fold_root = os.path.join(args_dict["data_path"], fold, args_dict["data_portion"]) 
    print(fold_root)
    LR = args_dict["learning_rate"]["lr_only"]
    patience = args_dict["learning_rate"]["patience"]
    gpu_no = (args_dict["computational_infra"]["fold_to_gpu_mapping"][fold])
    GPU = torch.device(f"cuda:{gpu_no}")
    pretraining_method = args_dict["method"]["name"]
    pair_sampling_method = args_dict["method"]["variant"]
    encoder = args_dict["encoder"]["name"]
    version = args_dict["encoder"]["version"]
    batch_size_list = args_dict["batch_size_list"]
    epochs = args_dict["epochs"]    
        
    # Get network for pretraining with MLP head
    model = None
    if "resnet" == args_dict["encoder"]["name"]:
        supervised_pretrained = False
        if "imagenet" == args_dict["encoder"]["pretrained"]:
            supervised_pretrained = True
        model = models.Resnet_SSL(
            version = version, 
            projector = args_dict["encoder"]["projector"],
            supervised_pretrained=supervised_pretrained)

    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(GPU)
    
    # Configure optimizer, schedular, loss, other configurations
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     factor=0.1,
                                                     patience=patience,
                                                     min_lr=5e-4)
    criterion = None
    if pretraining_method == "MPCS":
        criterion = ssl_loss.SimCLR_loss(gpu=GPU, temperature=args_dict["method"]["temperature"])
    
    
    
    experiment_description = f"_{fold}_{pretraining_method}_{pair_sampling_method}_{encoder}{version}_{args_dict['encoder']['pretrained']}_"

    # save expermental pramters details as yaml file in result folder with the name of experiment
    os.makedirs(os.path.join(args_dict["results"]["result_base_path"], experiment_description), exist_ok=True)
    with open(f"{os.path.join(args_dict['results']['result_base_path'], experiment_description)}/experiment_config.yaml", 'w') as file:
        documents = yaml.dump(args_dict, file)

    for batch_size in batch_size_list:
        
        # Load BreakHis dataset
        train_loader = datasets.get_BreakHis_trainset_loader(
            train_path=fold_root,
            training_method= pretraining_method,
            transform = transform.resize_transform,
            augmentation_strategy = augmentation_strategy.pretrain_augmentation,
            # Stain normalization
            pre_processing= [],
            image_pair=[40,100,200,400],
            pair_sampling_method = pair_sampling_method,
            batch_size = batch_size,
            num_workers = args_dict["computational_infra"]["workers"]
            )
        
        result_path =  os.path.join(args_dict["results"]["result_base_path"], experiment_description, str(batch_size))

        print(f" GPU {GPU} - Training - ", experiment_description, f" batch size {batch_size}")
        
        if pretraining_method == "MPCS":
            trainer = trainer_MPCS.Trainer_MPCS(
                experiment_description=experiment_description,
                pair_sampling_method = pair_sampling_method,
                dataloader=train_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=epochs,
                batch_size=batch_size,
                gpu = GPU,
                criterion=criterion,
                result_path=result_path,
                writer = SummaryWriter(log_dir=os.path.join(args_dict["logs"]["tensorboard_base_path"],experiment_description)),
                model_save_epochs_dir = args_dict["pretraining_model_saving_scheme"]
                )
            trainer.train()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MPCS Pre-training on BreakHis')
    parser.add_argument('--config', help='Config file for the experiment')
    args = parser.parse_args()
    args_dict = create_config(args.config)

    #mp.set_start_method('spawn')
    for fold in list(args_dict["computational_infra"]["fold_to_gpu_mapping"].keys()):
        #pretrain_model(args_dict, fold)
        process = mp.Process(target=pretrain_model, args=(args_dict, fold))
        process.start()
        #process.join()
