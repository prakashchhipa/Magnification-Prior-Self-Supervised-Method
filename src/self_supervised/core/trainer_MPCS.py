'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import argparse
import logging
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import os, copy, time
from pathlib import Path

from self_supervised.apply import config
from self_supervised.core import pretrain
from self_supervised.core import ssl_loss
from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Trainer_MPCS:
    def __init__(self,
                 experiment_description,
                 pair_sampling_method,
                 dataloader,
                 model,
                 optimizer,
                 scheduler,
                 epochs,
                 batch_size,
                 gpu,
                 criterion,
                 result_path,
                 writer,
                 model_save_epochs_dir):
        self.experiment_description = experiment_description
        self.pair_sampling_method = pair_sampling_method
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.previous_model = model
        self.current_model = model
        self.criterion = ssl_loss.SimCLR_loss(gpu=gpu, temperature=0.1)
        self.scheduler = scheduler
        self.gpu = gpu

        self.epochs = epochs
        self.current_epoch = 1
        self.lowest_loss = 10000
        self.cmd_logging = logging
        self.cmd_logging.basicConfig(level=logging.INFO,
                                     format='%(levelname)s: %(message)s')
        self.batch_size = batch_size
        self.loss_list = []
        self.loss_list.append(0)
        self.writer = writer
        self.input_images = []
        self.result_path = result_path
        self.model_save_epochs_dir = model_save_epochs_dir

    def train(self):
        for epoch in range(1, self.epochs + 1):

            self.current_epoch = epoch
            self.previous_model = self.current_model

            epoch_response_dir = pretrain.pretrain_epoch_MPCS(
                gpu = self.gpu,
                current_epoch=self.current_epoch,
                epochs=self.epochs,
                batch_size=self.batch_size,
                train_loader=self.dataloader,
                model=self.current_model,
                optimizer=self.optimizer,
                criterion=self.criterion)

            self.current_model = epoch_response_dir['model']
            self.loss_list.append(epoch_response_dir['loss'])
            logging.info(
                f' {self.gpu} {self.experiment_description} epoch: {epoch} MPCS {self.pair_sampling_method} loss: {self.loss_list[self.current_epoch]}'
            )
            self.input_images = epoch_response_dir['image_pair']

            #Logging tensor board
            self.tensorboard_analytics()

            #Save model - conditional
            self.save_model()

    def tensorboard_analytics(self):

        self.writer.add_scalar('SimCLR-Contrastive-Loss/Epoch',
                               self.loss_list[self.current_epoch],
                               self.current_epoch)

        self.writer.add_scalar('Learning_Rate/Epoch',
                               self.optimizer.param_groups[0]['lr'],
                               self.current_epoch)

        self.writer.add_image('View1/Aug',
                              self.input_images[0].detach().cpu().numpy()[0],
                              self.current_epoch)
        self.writer.add_image('View2/Aug',
                              self.input_images[1].detach().cpu().numpy()[0],
                              self.current_epoch)

    def save_model(self):
        if self.loss_list[self.current_epoch] < self.lowest_loss:
            intermediate_dir =None
            key_list = list(self.model_save_epochs_dir.keys())
            #100
            if self.current_epoch <= int(key_list[0]):
                intermediate_dir = self.model_save_epochs_dir[key_list[0]]
            #200
            elif self.current_epoch <= int(key_list[1]):
                intermediate_dir = self.model_save_epochs_dir[key_list[1]]
            #300
            elif self.current_epoch <= int(key_list[2]):
                intermediate_dir = self.model_save_epochs_dir[key_list[2]]
            #400
            elif self.current_epoch <= int(key_list[3]):
                intermediate_dir = self.model_save_epochs_dir[key_list[3]]
            #500
            elif self.current_epoch <= int(key_list[4]):
                intermediate_dir = self.model_save_epochs_dir[key_list[4]]
            #800
            elif self.current_epoch <= int(key_list[5]):
                intermediate_dir = self.model_save_epochs_dir[key_list[5]]
            #1000
            else:
                intermediate_dir = self.model_save_epochs_dir[key_list[6]]

            final_result_path = os.path.join(self.result_path, intermediate_dir)     
            os.makedirs(final_result_path, exist_ok=True)
            for file in Path(final_result_path).glob('*'):
                file.unlink()
            os.makedirs(final_result_path, exist_ok=True)
            torch.save(
                self.current_model.backbone.state_dict(),
                f"{final_result_path}/backbone_epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
            )
            torch.save(
                self.current_model.state_dict(),
                f"{final_result_path}/state_epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
            )
            self.lowest_loss = self.loss_list[self.current_epoch]

    def save_checkpoint(self):
        os.makedirs(
            f"{self.result_path+self.experiment_description}/checkpoints",
            exist_ok=True)
        torch.save(
            self.current_model.state_dict(),
            f"{self.result_path+self.experiment_description}/checkpoints/epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
        )
        
        
class Trainer_MPCS_Multi_GPU:
    def __init__(self,
                 experiment_description,
                 pair_sampling_method,
                 dataloader,
                 model,
                 optimizer,
                 optimizer_type,
                 scheduler,
                 sampler,
                 epochs,
                 batch_size,
                 gpu,
                 criterion,
                 result_path,
                 writer,
                 model_save_epochs_dir,
                 print_freq,
                 rank,
                 stats_file,
                 learning_rate_weights,
                 learning_rate_biases,
                 single_gpu,
                 current_epoch
                 ):
        
        self.experiment_description = experiment_description
        self.pair_sampling_method = pair_sampling_method
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.optimizer_type = optimizer_type
        self.previous_model = model
        self.current_model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.sampler = sampler
        self.gpu = gpu
        self.epochs = epochs
        self.current_epoch = current_epoch
        self.print_freq = print_freq
        self.stats_file = stats_file
        self.rank = rank
        self.learning_rate_weights = learning_rate_weights
        self.learning_rate_biases = learning_rate_biases
        self.single_gpu = single_gpu
        
        self.lowest_loss = 10000
        self.cmd_logging = logging
        self.cmd_logging.basicConfig(level=logging.INFO,
                                     format='%(levelname)s: %(message)s')
        self.batch_size = batch_size
        self.loss_list = []
        self.loss_list.append(0)
        self.writer = writer
        self.input_images = []
        self.result_path = result_path
        self.model_save_epochs_dir = model_save_epochs_dir
        
    def train(self):
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(1, self.epochs + 1):

            self.current_epoch = epoch
            self.previous_model = self.current_model
            if False == self.single_gpu:
                self.sampler.set_epoch(epoch)

            epoch_response_dir = pretrain.pretrain_epoch_MPCS_multi_gpu(
                gpu = self.gpu,
                current_epoch=self.current_epoch,
                epochs=self.epochs,
                batch_size=self.batch_size,
                train_loader=self.dataloader,
                model=self.current_model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scaler=scaler,
                learning_rate_weights=self.learning_rate_weights,
                learning_rate_biases=self.learning_rate_biases,
                print_freq=self.print_freq,
                stats_file=self.stats_file,
                start_time=start_time,
                rank=self.rank,
                optimizer_type = self.optimizer_type,
                single_gpu=self.single_gpu
                )

            self.current_model = epoch_response_dir['model']
            self.backbone = epoch_response_dir['backbone']
            self.state = epoch_response_dir['state']
            self.loss_list.insert(self.current_epoch,
                                  epoch_response_dir['loss'])
            logging.info(
                f' {self.gpu} {self.experiment_description} epoch: {epoch} MPCS {self.pair_sampling_method} loss: {self.loss_list[self.current_epoch]}'
            )
            self.input_images = epoch_response_dir['image_pair']

            #Logging tensor board
            self.tensorboard_analytics()

            #Save model - conditional
            if 0 == self.rank:
                self.save_model()

    def tensorboard_analytics(self):

        self.writer.add_scalar('SimCLR-Contrastive-Loss/Epoch',
                               self.loss_list[self.current_epoch],
                               self.current_epoch)

        self.writer.add_scalar('Learning_Rate/Epoch',
                               self.optimizer.param_groups[0]['lr'],
                               self.current_epoch)

        self.writer.add_image('View1/Aug',
                              self.input_images[0].detach().cpu().numpy()[0],
                              self.current_epoch)
        self.writer.add_image('View2/Aug',
                              self.input_images[1].detach().cpu().numpy()[0],
                              self.current_epoch)

    def save_model(self):
        if self.loss_list[self.current_epoch] < self.lowest_loss:
            intermediate_dir =None
            key_list = list(self.model_save_epochs_dir.keys())
            #100
            if self.current_epoch <= int(key_list[0]):
                intermediate_dir = self.model_save_epochs_dir[key_list[0]]
            #200
            elif self.current_epoch <= int(key_list[1]):
                intermediate_dir = self.model_save_epochs_dir[key_list[1]]
            #300
            elif self.current_epoch <= int(key_list[2]):
                intermediate_dir = self.model_save_epochs_dir[key_list[2]]
            #400
            elif self.current_epoch <= int(key_list[3]):
                intermediate_dir = self.model_save_epochs_dir[key_list[3]]
            #500
            elif self.current_epoch <= int(key_list[4]):
                intermediate_dir = self.model_save_epochs_dir[key_list[4]]
            #800
            elif self.current_epoch <= int(key_list[5]):
                intermediate_dir = self.model_save_epochs_dir[key_list[5]]
            #1000
            else:
                intermediate_dir = self.model_save_epochs_dir[key_list[6]]

            final_result_path = os.path.join(self.result_path, intermediate_dir)     
            os.makedirs(final_result_path, exist_ok=True)
            for file in Path(final_result_path).glob('*'):
                if 'nfs' not in str(file): 
                    file.unlink()
            os.makedirs(final_result_path, exist_ok=True)
            torch.save(
                self.backbone,
                f"{final_result_path}/backbone_epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
            )
            torch.save(
                self.state,
                f"{final_result_path}/state_epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
            )
            self.lowest_loss = self.loss_list[self.current_epoch]

    def save_checkpoint(self):
        os.makedirs(
            f"{self.result_path+self.experiment_description}/checkpoints",
            exist_ok=True)
        torch.save(
            self.current_model.state_dict(),
            f"{self.result_path+self.experiment_description}/checkpoints/epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
        )