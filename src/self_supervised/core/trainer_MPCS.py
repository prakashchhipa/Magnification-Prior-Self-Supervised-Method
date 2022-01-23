'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import argparse
import logging
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import os, copy
from pathlib import Path

from self_supervised.apply import config
from self_supervised.core import pretrain
from self_supervised.core import ssl_loss
from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#Trainer - Magnification Prior Contrastive Similarity Method
class Trainer_MPCS:
    def __init__(self,
                 experiment_description,
                 dataloader,
                 model,
                 optimizer,
                 scheduler,
                 epochs,
                 batch_size,
                 gpu,
                 criterion):
        self.experiment_description = experiment_description
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.previous_model = model
        self.current_model = model
        self.criterion = ssl_loss.SimCLR_loss(gpu=gpu, temperature=0.1)
        self.scheduler = scheduler
        self.gpu = gpu

        self.epochs = epochs
        self.current_epoch = 11
        self.lowest_loss = 10000
        self.cmd_logging = logging
        self.cmd_logging.basicConfig(level=logging.INFO,
                                     format='%(levelname)s: %(message)s')
        self.batch_size = batch_size
        self.loss_list = []
        self.loss_list.append(0)
        self.writer = SummaryWriter(log_dir=config.tensorboard_path+experiment_description)
        self.input_images = []

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
                f'{self.experiment_description} epoch: {epoch} simCLR loss: {self.loss_list[self.current_epoch]}'
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
            os.makedirs(config.result_path +self.experiment_description,
                        exist_ok=True)
            torch.save(
                self.current_model.state_dict(),
                f"{config.result_path+self.experiment_description}/epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
            )
            self.lowest_loss = self.loss_list[self.current_epoch]

    def save_checkpoint(self):
        os.makedirs(
            f"{config.result_path+self.experiment_description}/checkpoints",
            exist_ok=True)
        torch.save(
            self.current_model.state_dict(),
            f"{config.result_path+self.experiment_description}/checkpoints/epoch_{self.current_epoch}_loss_{self.loss_list[self.current_epoch]}.pth"
        )