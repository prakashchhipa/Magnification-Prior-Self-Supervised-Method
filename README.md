# Magnification-Prior-Self-Supervised-Method

Implementation for 'Magnification Prior: A Self-Supervised Method for Learning Representations on Breast Cancer Histopathological Images'

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magnification-prior-a-self-supervised-method/breast-cancer-histology-image-classification)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification?p=magnification-prior-a-self-supervised-method)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magnification-prior-a-self-supervised-method/breast-cancer-histology-image-classification-1)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification-1?p=magnification-prior-a-self-supervised-method)


# Requirement
This repository code is compaitible with Python 3.6 and 3.8, Pytorch 1.2.0, and Torchvision 0.4.0.

# Dataset
BreakHis is publically available dataset on Breast Cancer Histopathology WSI of several magnifications. Link - https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Details from BreakHis website:
The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).  To date, it contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). This database has been built in collaboration with the P&D Laboratory  – Pathological Anatomy and Cytopathology, Parana, Brazil (http://www.prevencaoediagnose.com.br). We believe that researchers will find this database a useful tool since it makes future benchmarking and evaluation possible.

# Commands

**Self-supervised pretraining (Assuming in directory 'src')** 

```python -m self_supervised.experiments.pretrain_MPCS --data_fold <'train_data_fold_path'> --pair_sampling <'OP'/'RP'/'FP'> --LR <learning_rate - 0.00001> --epoch <150> --description <'experiment_name'>```
**OP - Ordered Pair, RP - Random Pair, and FP - Fixed Pair

**Fintuning using ImageNet pretrained Efficient-net b2 on BreakHis (Assuming in directory 'src')**

```python -m supervised.experiments.finetune_imagenet_on_breakhis --train_data_fold <'train_data_fold_path'> --test_data_fold <'test_data_fold_path'> --magnification <'40x'/'100x'/'200x'/'400x'> --LR <learning_rate - 0.00002> --epoch <150> --description <'experiment_name'>```

**Fintuning using MPCS pretrained Efficient-net b2 on BreakHis (Assuming in directory 'src')**

```python -m supervised.experiments.finetune_mpcs_on_breakhis --train_data_fold <'train_data_fold_path'> --test_data_fold <'test_data_fold_path'> --magnification <'40x'/'100x'/'200x'/'400x'>  --model_path <'pretrained model path'> --LR <learning_rate - 0.00002> --epoch <150> --description <'experiment_name'>```

**Evaluation**

```python - m supervised.evaluation.evaluation <check paramters in file itself>```

