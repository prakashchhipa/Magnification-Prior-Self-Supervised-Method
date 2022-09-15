'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import argparse
import logging
import os, sys, yaml
from cv2 import split
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from supervised.apply.datasets import get_BreakHis_data_loader,get_BreakHis_testdata_loader 
from supervised.apply.transform import train_transform, resize_transform
from self_supervised.core import ssl_loss, models, pretrain, utility, trainer_MPCS
sys.path.append(os.path.dirname(__file__))
from self_supervised.apply import datasets, config, transform, augmentation_strategy
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_WARNINGS"] = "FALSE"

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time


import bc_config


def get_embeddings(fold_root, portion, model_path, magnification):

    test_loader = get_BreakHis_testdata_loader(
        fold_root + f'{portion}/', 
        transform = resize_transform,
        pre_processing=[], 
        image_type_list= [magnification],
        batch_size=32
        )

    projector = "1024-128"
    version = 50
    device = torch.device("cuda:7") #change according to GPU device availability

    model = models.Dilated_Resnet_SSL(version = version, projector = projector, supervised_pretrained=False).cuda(device)
    checkpoint = torch.load(model_path)["model"]
    checkpoint_new = {}
    for key in checkpoint.keys():
        key_splits = key.split(".")
        key_ = ""
        for idx in range(1,len(key_splits)-1):
            key_ += key_splits[idx]
            key_ += "."
        key_ += key_splits[len(key_splits)-1]
        checkpoint_new[key_] = checkpoint[key]
    model.load_state_dict(checkpoint_new)

    embeddings =[]
    labels =[]

    model.eval()
    with torch.no_grad():
        for patient_id, magnification, item_dict, binary_label, multi_label in tqdm(test_loader):
            view = item_dict[magnification[0]]
            view = view.cuda(device, non_blocking=True)                
            
            outputs = model(view)
            
            embeddings.append(outputs.cpu())
            labels.append(binary_label)
            
    embeddings = torch.cat(embeddings, dim = 0)
    labels = torch.cat(labels, dim = 0)
    embeddings = embeddings.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    print("embedding generated")
    
    return embeddings, labels

def get_tsne(data, n_components = 2, n_images = None):
        
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    print(tsne_data)
    return tsne_data

def plot_representations(file_name,data, labels, n_images = None):
            
    print("data len- ", len(data))
    print("label len", len(labels))
    x = data[:, 0]
    y = data[:, 1]
    labels = labels.astype('uint8')                
    unique = list(set(labels))
    legend_dict = {0: "bengin", 1: "malignant"}
    colors = [plt.cm.coolwarm(float(i)/max(unique)) for i in unique]
    
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    for i, u in enumerate(unique):
        xi = [x[j] for j  in range(len(x)) if labels[j] == u]
        yi = [y[j] for j  in range(len(x)) if labels[j] == u]
        ax.scatter(xi, yi, c=colors[i], label=str(legend_dict[u]), cmap = 'viridis')
    ax.legend()
    
    plt.savefig(f'/home/output/_{file_name}.png')

if __name__ == "__main__":
    
    model_path_dict =  {
        "op" : "", #give actual model full path
        "rp" : "", #give actual model full path
        "fp" : ""  #give actual model full path
    }
    
    data_path_dict = {
        "0" : "/home/datasets/BreaKHis_v1/histology_slides/breast/Fold_0_5/",
        "1" : "/home/datasets/BreaKHis_v1/histology_slides/breast/Fold_1_5/",
        "2" : "/home/datasets/BreaKHis_v1/histology_slides/breast/Fold_2_5/",
        "3" : "/home/datasets/BreaKHis_v1/histology_slides/breast/Fold_3_5/",
        "4" : "/home/datasets/BreaKHis_v1/histology_slides/breast/Fold_4_5/"
    }
   
    portion_list = ["test_20"]
    
    magnification_dict = {
        "40x" : bc_config.X40,
        "100x" : bc_config.X100,
        "200x" : bc_config.X200,
        "400x" : bc_config.X400
    }
    
    for model_key in model_path_dict.keys():
        for data_key in data_path_dict.keys():
            for portion in portion_list:
                for mag_key in magnification_dict.keys():
                    embeddings, labels = get_embeddings(data_path_dict[data_key], portion, model_path_dict[model_key], magnification_dict[mag_key])
                    output_data = get_tsne(embeddings)
                    file_name = f"{mag_key}_{model_key}_{data_key}"
                    print(file_name)
                    plot_representations(file_name, output_data, labels)
                    print(f"t-SNE for {file_name} saved")
    
    




