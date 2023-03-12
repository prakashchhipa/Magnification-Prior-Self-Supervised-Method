# Magnification-Prior-Self-Supervised-Method

Implementation for ['Magnification Prior: A Self-Supervised Method for Learning Representations on Breast Cancer Histopathological Images'](https://openaccess.thecvf.com/content/WACV2023/html/Chhipa_Magnification_Prior_A_Self-Supervised_Method_for_Learning_Representations_on_Breast_WACV_2023_paper.html) - Accepted in EEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023

Chhipa, P. C., Upadhyay, R., Pihlgren, G. G., Saini, R., Uchida, S., & Liwicki, M. (2023). Magnification prior: a self-supervised method for learning representations on breast cancer histopathological images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2717-2727).


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magnification-prior-a-self-supervised-method/breast-cancer-histology-image-classification)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification?p=magnification-prior-a-self-supervised-method)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magnification-prior-a-self-supervised-method/breast-cancer-histology-image-classification-1)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification-1?p=magnification-prior-a-self-supervised-method)


# Requirement
This repository code is compaitible with Python 3.6 and 3.8, Pytorch 1.2.0, and Torchvision 0.4.0.

# Datasets
1. **BreakHis** - This is publically available dataset on Breast Cancer Histopathology WSI of several magnifications. Link - https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/. Details from BreakHis website: The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).  To date, it contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). This database has been built in collaboration with the P&D Laboratory  – Pathological Anatomy and Cytopathology, Parana, Brazil (http://www.prevencaoediagnose.com.br). We believe that researchers will find this database a useful tool since it makes future benchmarking and evaluation possible.

2. **BACH** - The second dataset, Breast Cancer Histology Images (BACH) [2] is publically available from the ICIAR2018 Grand Challenge and contains 400 histopathology slides. The BACH dataset has four classes, normal, benign, in-situ, and invasive. The slide size is relatively large, 2048 × 1536 pixels; thus, patches of size 512x512. This dataset can be access via https://iciar2018-challenge.grand-challenge.org/Dataset/.

3. **Breast Cancer Cell Dataset** - The third publically available dataset, Breast Cancer Cell Dataset is from the University of California, Santa Barbara Biosegmentation Benchmark. This dataset contains 58 HE-stained histopathology 896x768 size images of breast tissue, of which 26 are malignant, and 32 are benign. This dataset can be access via https://bioimage.ucsb.edu/research/bio-segmentation


# Commands

**Data access, prepartion, and processing scripts in src/data package**

**1. BreakHis dataset** 
```python -m prepare_data_breakhis```
**2. BACH dataset** 
```python -m prepare_data_bach```
```python -m prepare_metadata_bach```
```python -m stain_norm_bach_data```
```python -m prepare_augmented_patches_bach```
```python -m create_data_portion_for_augmented_patches_bach```
**3. Breast Cancer Cell dataset** 
```python -m prepare_data_bisque```
```python -m prepare_metadata_bisque```
**Choose random seed for each dataset preaprtion - experiments were condcuted using three seeds 47, 86, 16, 12

**Self-supervised pretraining on BreakHis Dataset** 

**1. Single GPU implementation for constrained computation - use and customize the config files located in src/self_supervised/experiment_config/single_gpu - example mentioned below** 
```python -m pretrain_mpcs_single_gpu --config experiment_config/single_gpu/mpcs_op_rn50.yaml```
**It choses Ordered Pair smapling method for MPCS pretraining for ResNet50 encoder. Refer config files for cokmplete details and alternatives. Batch size needs to be small in this settings.

**2. Multi GPU implementation for large batch size support - use and customize the config files located in src/self_supervised/experiment_config/multi_gpu - example mentioned below** 
```python -m pretrain_mpcs_multi_gpu --config experiment_config/multi_gpu/mpcs_op_rn50.yaml```
**It choses Ordered Pair smapling method for MPCS pretraining for ResNet50 encoder. Refer config files for cokmplete details and alternatives. It can support any batch size for pretraining given sufficient computation nodes.

**Downstream Task on BreakHis dataset** 
**1. ImageNet supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_breakhis --config experiment_config/breakhis_imagenet_rn50.yaml```
**Refer config files for cokmplete details and alternatives. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Evaluation takes place after finetununbg completed on validation and testset and results are logged. no manual instruction needed.

**2. MPCS self-supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_breakhis --config experiment_config/breakhis_mpcs_rn50.yaml```
**Refer config files for cokmplete details and alternatives and smapling method ordered pair, fixed pair and random pair. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Pretraine models are search, accessed by scripts for given base path of all models autonomously and it fine tune models for each listed pretrained model weights for each batch size available. Evaluation takes place after finetuning completed on validation and testset and results are logged. no manual instruction needed.

**Downstream Task on BACH dataset** 
**1. ImageNet supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_bach --config experiment_config/bach_imagenet_rn50_data100.yaml```
**Refer config files for cokmplete details and alternatives. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Evaluation takes place after finetununbg completed on testset and results are logged. no manual instruction needed.

**2. MPCS self-supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_bach --config experiment_config/bach_mpcs_op_dilated_rn50_1024_100_data100_224.yaml```
**Refer config files for cokmplete details and alternatives and smapling method ordered pair, fixed pair and random pair. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Pretraine models are search, accessed by scripts for given base path of all models autonomously and it fine tune models for each listed pretrained model weights for each batch size available. Evaluation takes place after finetuning completed on testset and results are logged. no manual instruction needed.

**Downstream Task on Breat Cancer Cell  dataset** 

**1. MPCS self-supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_bisque --config experiment_config/bisque_mpcs_fp_dilated_rn50_1024_100_data100_224.yaml```
**Refer config files for cokmplete details and alternatives and smapling method ordered pair, fixed pair and random pair. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Pretraine models are search, accessed by scripts for given base path of all models autonomously and it fine tune models for each listed pretrained model weights for each batch size available. Evaluation takes place after finetuning completed on testset and results are logged. no manual instruction needed.

**2. MPCS self-supervised transfer learning linear evaluation for malgnancy classification** 
```python -m linear_eval_bisque --config experiment_config/bisque_mpcs_fp_dilated_rn50_1024_100_data100_224.yaml```
**Refer config files for cokmplete details and alternatives and smapling method ordered pair, fixed pair and random pair. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Pretraine models are search, accessed by scripts for given base path of all models autonomously and it fine tune models for each listed pretrained model weights for each batch size available. Evaluation takes place after finetuning completed on testset and results are logged. no manual instruction needed.

**Exaplainable results - class actviation maps** 

```python class_activation_map-ipynb```

**Evaluation - however evaluaiton is covered in above mentioned scripts but it can be perofrmed externally using following script 

```python -m evaluation```

