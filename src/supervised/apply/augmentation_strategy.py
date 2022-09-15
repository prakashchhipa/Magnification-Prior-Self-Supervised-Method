'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from self_supervised.apply import config

ft_augmentation = A.Compose([
        A.Flip(p=0.2),
        A.Rotate(p=0.2),
        A.Affine(translate_percent = 0.05, p=0.2),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])


augmentation_bach_03 = A.Compose([
        A.ColorJitter (brightness=0.2, contrast=0.2, always_apply=False, p=0.3),
        A.RandomRotate90(p=1)
        ])


augmentation_bach_08 = A.Compose([
        A.ColorJitter (brightness=0.4, contrast=0.2,saturation=0.1, hue=0.1, always_apply=False, p=0.8),
        A.ColorJitter (brightness=0.2, contrast=0.4,saturation=0.1, hue=0.1, always_apply=False, p=0.8),
        A.RandomRotate90(p=1)
        ])



augmentation_03 = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.3),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])

augmentation_05 = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.6),
        A.Flip(p=0.5),
        A.Rotate(p=0.5),
        A.Affine(translate_percent = 0.1, p=0.5),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])

augmentation_08 = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.8),
        A.Flip(p=0.9),
        A.Rotate(p=0.9),
        A.Affine(translate_percent = 0.1, p=0.8)
        ])

ft_rcfor1_exp_augmentation = A.Compose([
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Resize(height=230, width=350, p=1),
        A.RandomCrop(height=230,width=300,p=1)
        ])
