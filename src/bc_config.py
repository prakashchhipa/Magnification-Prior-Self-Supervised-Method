'''Author- Prkaash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import torch

batch_size = 14
num_workers = 1


#Binary labels
binary_label_B = 'B'
binary_label_M = 'M'

#Multi labels
multi_label_A = 'A'
multi_label_F = 'F'
multi_label_PT = 'PT'
multi_label_TA = 'TA'
multi_label_DC = 'DC'
multi_label_MC = 'MC'
multi_label_LC = 'LC'
multi_label_PC = 'PC'

bach_label_insitu = 'insitu'
bach_label_normal = 'normal'
bach_label_invasive = 'invasive'
bach_label_benign = 'benign'



binary_label_dict = {binary_label_B: 0, binary_label_M: 1}
multi_label_dict = {multi_label_A:0,multi_label_F:1,multi_label_PT:2,multi_label_TA:3,multi_label_DC:4,multi_label_MC:5,multi_label_LC:6,multi_label_PC:7}

binary_label_list = [binary_label_B,binary_label_M]
multi_label_list = [multi_label_A,multi_label_F,multi_label_PT,multi_label_TA,multi_label_DC,multi_label_MC,multi_label_LC,multi_label_PC]

bach_label_dict = {bach_label_benign:0,bach_label_invasive:1,bach_label_insitu:2,bach_label_normal:3}
bach_label_list = [bach_label_benign,bach_label_invasive,bach_label_insitu,bach_label_normal]


data_path_fold0 = '/home/datasets/breast/Fold_0_5/'
data_path_fold1 = '/home/datasets/breast/Fold_1_5/'
data_path_fold2 = '/home/datasets/breast/Fold_2_5/'
data_path_fold3 = '/home/datasets/breast/Fold_3_5/'
data_path_fold4 = '/home/datasets/breast/Fold_4_5/'

result_path = '/home/result/results_bc_5fold/'
tensorboard_path = '/home/logs/tensorboard_bc_5fold/'

#GPU
gpu0 = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")
gpu2 = torch.device("cuda:2")
gpu3 = torch.device("cuda:3")
gpu4 = torch.device("cuda:4")
gpu5 = torch.device("cuda:5")
gpu6 = torch.device("cuda:6")
gpu7 = torch.device("cuda:7")

gpu_device_dict = {0: gpu0, 1: gpu1, 2: gpu2,
                   3: gpu3, 4: gpu4, 5: gpu5, 6: gpu6, 7: gpu7}

#magnification
X40 = '40X'
X100 = '100X' 
X200 = '200X'
X400 = '400X'

#dataset portion
train = 'train'
test = 'test'
val = 'val'

#networks
EfficientNet_b2 = 'EfficientNet_b2'
DenseNet_121 = 'DenseNet_121'
Resnet_150 = 'Resnet_150'
Resnet_50 = 'Resnet_50'

#Model params
num_classes = 2

#Stain Noramlization
Reinhard_Normalization = 'Reinhard2001'
Macenko_Normalization = 'Macenko2009'
Vahadane_Normalization = 'Vahadane2015'

#valset  split for SSL finetuning into train and validation
portion_train = 'val_train'
portion_val = 'val_val'