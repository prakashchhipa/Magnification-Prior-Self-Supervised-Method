'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import argparse
from base64 import encode
import logging
import os, sys, yaml

import sys
import time
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from self_supervised.core import ssl_loss, models, optimizers, pretrain, utility, trainer_MPCS
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

def main():
    parser = argparse.ArgumentParser(description='BreakHis self-supervised method Pre-training')
    parser.add_argument('--config', help='Config file for the experiment')
    args = parser.parse_args()
    args_dict = create_config(args.config)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args_dict['computational_infra']['allowed_gpus']
    os.environ['NUMEXPR_MAX_THREADS'] = args_dict['computational_infra']['numexpr_num_threads']

    args_dict['distributed_processing']['ngpus_per_node'] = torch.cuda.device_count()
    args_dict['single_gpu'] = False
    if args_dict['distributed_processing']['ngpus_per_node'] == 1:
        logging.info(
            '!...........Only single GPU found so running on single GPU mode...........!')
        args_dict['single_gpu'] = True
    logging.info(
        f'ngpus_per_node {args_dict["distributed_processing"]["ngpus_per_node"]}')
    # single-node distributed training
    args_dict['distributed_processing']['rank'] = 0
    args_dict['distributed_processing']['world_size'] = args_dict['distributed_processing']['ngpus_per_node']
    torch.multiprocessing.spawn(
        main_worker, (args_dict,), args_dict['distributed_processing']['ngpus_per_node'])


def main_worker(gpu, args_dict):
    
    #START: Program Input**********************************************************************************************************
    
    #1. data
    data_path = args_dict["data"]["data_path"]
    data_portion = args_dict["data"]["data_portion"]
    
    #2. encoder
    encoder = args_dict["encoder"]["name"]
    version = args_dict["encoder"]["version"]
    projector = args_dict["encoder"]["projector"]
    pretrained_type = args_dict["encoder"]["pretrained"]
    weights_initialization_enable = args_dict["encoder"]["weights_initialization"]["enable"]
    weights_initialization_checkpoint_path = args_dict["encoder"]["weights_initialization"]["checkpoint_path"]
    resume_training_enable = args_dict["encoder"]["resume_training"]["enable"]
    resume_training_checkpoint_path = args_dict["encoder"]["resume_training"]["checkpoint_path"]
    
    #3. ssl method
    pretraining_method = args_dict["ssl_method"]["name"]
    pair_sampling_method = args_dict["ssl_method"]["variant"]
    temperature=args_dict["ssl_method"]["temperature"]
    
    #4. training parameters
    batch_size_list = args_dict["training_parameters"]["batch_size_list"]
    epochs = args_dict["training_parameters"]["epochs"]
    optimizer_type = args_dict["training_parameters"]["optimizer"]
    # specifically for for Adam
    LR = args_dict["training_parameters"]["learning_rate"]["lr_only"]
    patience = args_dict["training_parameters"]["learning_rate"]["patience"]
    weight_decay = args_dict["training_parameters"]["learning_rate"]["weight_decay"]
    #for LARS
    lars_lr=args_dict["training_parameters"]["lars_optimizer"]["lr"]
    lars_weight_decay=args_dict["training_parameters"]["lars_optimizer"]["weight_decay"]
    lars_momentum=args_dict["training_parameters"]["lars_optimizer"]["momentum"]
    lars_eta=args_dict["training_parameters"]["lars_optimizer"]["eta"]
    lars_weight_decay_filter=args_dict["training_parameters"]["lars_optimizer"]["weight_decay_filter"]
    lars_adaptation_filter=args_dict["training_parameters"]["lars_optimizer"]["lars_adaptation_filter"]
    lars_learning_rate_weights = args_dict["training_parameters"]["lars_optimizer"]["learning_rate_weights"]
    lars_learning_rate_biases = args_dict["training_parameters"]["lars_optimizer"]["learning_rate_biases"]
    
    #5. utilities
    pretraining_model_saving_scheme_dir = args_dict["utility"]["pretraining_model_saving_scheme"]
    print_freq = args_dict["utility"]["print_freq"]
    
    #6. logging & results
    tensorboard_base_path = args_dict["logs"]["tensorboard_base_path"]
    #tensorboard_file_path = args_dict["logs"]["tensorboard_file_path"] #dymanically updated
    #stats_file_path = args_dict["logs"]["stats_file_path"] #dymanically updated
    result_base_path = args_dict["results"]["result_base_path"]
    #result_dir_path = args_dict["results"]["result_dir_path"] #dymanically updated
    
    
    #7. computations
    allowed_gpus = args_dict["computational_infra"]["allowed_gpus"]
    workers = args_dict["computational_infra"]["workers"]
    numexpr_num_threads = args_dict["computational_infra"]["numexpr_num_threads"]
    cudnn_benchmark = args_dict["computational_infra"]["cudnn_benchmark"]
    single_gpu = args_dict["single_gpu"] #dymanically created in main function based on GPU counts
    
    #8. distributed processing support
    backend = args_dict['distributed_processing']['backend']
    init_method=args_dict['distributed_processing']['dist_url']
    world_size = args_dict['distributed_processing']['world_size']
    rank = args_dict['distributed_processing']['rank']
    
    #STOP: Program Input**********************************************************************************************************
    
    #GPU selection and setting up multi-spu configuration support
    rank += gpu
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )

    #Experiment Description (also used as top level directory to save results and logs)
    detailed_experiment_description = f'_{encoder}{version}_{pretraining_method}_{pair_sampling_method}_WI_{pretrained_type}_OP_{optimizer_type}_DP{data_portion}_EP{epochs}_'
        
    #Results and Logging paths
    tb_log_dir = os.path.join(tensorboard_base_path, detailed_experiment_description)
    args_dict["logs"]["tensorboard_file_path"] = tb_log_dir #--> saved in experiment description xml file
    writer = SummaryWriter(log_dir=tb_log_dir)
    result_path = os.path.join(result_base_path, f'{encoder}',  f'{detailed_experiment_description}')
    args_dict["results"]["result_dir_path"] = result_path #--> saved in experiment description xml file
    stats_file_path = os.path.join(result_path, 'stats.txt') #--> fix name for file under different experiment description directories
    if args_dict['distributed_processing']['rank'] == 0: #--> stat file is created by only first thread (which access first GPU)
        os.makedirs(result_path, exist_ok=True)
        stats_file = open(stats_file_path, 'a', buffering=1)
        args_dict["logs"]["stats_file_path"] = stats_file_path
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    logging.info(f'gpu - {bc_config.gpu_device_dict[gpu]} used by {detailed_experiment_description}')
    
    #GPU assignement as part of training
    args_dict['computational_infra']["gpu"] = bc_config.gpu_device_dict[gpu]
    torch.cuda.set_device(bc_config.gpu_device_dict[gpu])
    torch.backends.cudnn.benchmark = args_dict["computational_infra"]["cudnn_benchmark"]
    
    #Encoder(model) architecture selection for pretraining including MLP head
    model = None
    if "resnet" == encoder:
        supervised_pretrained = False
        if "imagenet" == pretrained_type:
            supervised_pretrained = True
        model = models.Resnet_SSL(version = version, projector = projector, supervised_pretrained=supervised_pretrained).cuda(bc_config.gpu_device_dict[gpu])
    elif "dilated_resnet" == encoder:
        supervised_pretrained = False
        if "imagenet" == pretrained_type:
            supervised_pretrained = True
        model = models.Dilated_Resnet_SSL(version = version, projector = projector, supervised_pretrained=supervised_pretrained).cuda(bc_config.gpu_device_dict[gpu])
        print(model)
    elif "efficientnet" == encoder:
        supervised_pretrained = False
        if "imagenet" == pretrained_type:
            supervised_pretrained = True
        model = models.EfficientNet_SSL(version=str(version), projector=projector, supervised_pretrained=supervised_pretrained).cuda(bc_config.gpu_device_dict[gpu])
    else:
        raise ValueError("Input enncoder is not implemented for self-supervised pretraining")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if False == single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[bc_config.gpu_device_dict[gpu]]) #-> enabling model parallelism if more than one GPU available in world_size
        
    #Encoder to initilialize with specific weights
    resume_epoch = 1 # Default training starts with epoch no 1
    if "imagenet" != pretrained_type and True == weights_initialization_enable:
        if len(weights_initialization_checkpoint_path) > 0:
            logging.info(
                f'START - Loading SimCLR Imagenet pretrained model for encoder RN{str(args_dict["resnet_arch"])} and projector layers')
            model.backbone.load_state_dict(torch.load(
                args_dict["checkpoint_path"]), strict=False)
            logging.info(
                f'STOP - Loading SimCLR Imagenet pretrained model for encoder RN{str(args_dict["resnet_arch"])} and projector layers')
        else:
            raise ValueError(
                'Weights initialization is enabled but ["weights_initialization"]["checkpoint_path"] is not valid')

    if "imagenet" != pretrained_type and True == resume_training_enable:
        checkpoint_path = resume_training_checkpoint_path
        if len(checkpoint_path) > 0:
            # 1. resume-training preprocessing logic to get last epoch until model was trained
            resume_epoch = int(checkpoint_path.split("/")[-1].split("_")[3])
            logging.info(
                f'START - Loading saved model from epoch {resume_epoch} for encoder RN{str(args_dict["resnet_arch"])} and projector layers')
            # 2. load model
            model.load_state_dict(torch.load(checkpoint_path)[
                                  "model"], strict=False)
            logging.info(
                f'STOP - Loading saved model from epoch {resume_epoch} for encoder RN{str(args_dict["resnet_arch"])} and projector layers')
        else:
            raise ValueError(
                'resume training is enabled but ["resume_training"]["checkpoint_path"] is not valid')

    #MPCS - SimCLR based loss for criterion
    criterion = None
    if "MPCS" == pretraining_method:
        criterion = ssl_loss.SimCLR_loss_multi_GPU(batch_size=batch_size_list[0], temperature=temperature, world_size=world_size)
    elif "MPSN" == pretraining_method:
        raise ValueError("Loss function for MPSN in distributed setting yet to be implemented")
    else:
        raise ValueError("Input method not available or incorrect")
    
    #Optimizer choices
    optimizer = None
    scheduler = None
    # 1. LARS
    if "LARS" == optimizer_type:
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = optimizers.LARS(
            parameters, lr=lars_lr,
            weight_decay=lars_weight_decay,
            momentum=lars_momentum,
            eta=lars_eta,
            weight_decay_filter=lars_weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
    # 2. Adam
    elif "adam" == optimizer_type:
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        'min',
                                                        factor=0.1,
                                                        patience=patience,
                                                        min_lr=5e-4)
    else:
        raise ValueError("Input optimizer type is not implemented yet or incorrect")

    #Self-supervised pretraining method choice
    if pair_sampling_method not in ["OP", "FP", "RP"]:
        raise ValueError("Input smapling method is incorrect")
        
    #Batchwise - pretraining 
    for batch_size in batch_size_list:
        
        # Data & loaders
        train_loader = None
        sampler = None
        if True == single_gpu:
            train_loader = datasets.get_BreakHis_trainset_loader(
                train_path= os.path.join(data_path, data_portion),
                training_method= pretraining_method,
                transform = transform.resize_transform_224,
                augmentation_strategy = augmentation_strategy.pretrain_augmentation,
                # Stain normalization
                pre_processing= [],
                image_pair=[40,100,200,400],
                pair_sampling_method = pair_sampling_method,
                batch_size = batch_size,
                num_workers = workers
                )
        else:
            train_loader, sampler = datasets.get_BreakHis_trainset_loader_multi_gpu(
                train_path= os.path.join(data_path, data_portion),
                training_method= pretraining_method,
                transform = transform.resize_transform_224,
                augmentation_strategy = augmentation_strategy.pretrain_augmentation,
                # Stain normalization
                pre_processing= [],
                image_pair=[40,100,200,400],
                pair_sampling_method = pair_sampling_method,
                batch_size = batch_size,
                num_workers = workers,
                world_size = world_size
                )
            
        #Logging experiment description
        stats_file_ref = None #-> to be used as arugment for trainer class
        if 0 == rank:
            stats_file_ref = stats_file
            logging.info(
                f'Experiment derscription - {detailed_experiment_description}')
            # save expermental parameters details as yaml file in result folder with the name of experiment
            with open(f"{result_path}/experiment_config.yaml", 'w') as file:
                documents = yaml.dump(args_dict, file)
        
        batch_wise_result_path =  os.path.join(result_path, detailed_experiment_description, str(batch_size))

        if pretraining_method == "MPCS":
            trainer = trainer_MPCS.Trainer_MPCS_Multi_GPU(
                experiment_description=detailed_experiment_description,
                pair_sampling_method = pair_sampling_method,
                dataloader=train_loader,
                model=model,
                optimizer=optimizer,
                optimizer_type = optimizer_type,
                scheduler=scheduler,
                sampler = sampler,
                epochs=epochs,
                batch_size=batch_size,
                gpu = gpu,
                criterion=criterion,
                result_path=batch_wise_result_path,
                model_save_epochs_dir = pretraining_model_saving_scheme_dir,
                print_freq=print_freq,
                rank=rank,
                stats_file=stats_file_ref,
                #lars specific
                learning_rate_weights=lars_learning_rate_weights,
                learning_rate_biases=lars_learning_rate_biases,
                writer=writer,
                single_gpu=single_gpu,
                # resume training from next epoch from the model was saved
                current_epoch=resume_epoch + 1
                )
            trainer.train()




if __name__ == '__main__':

    main()
