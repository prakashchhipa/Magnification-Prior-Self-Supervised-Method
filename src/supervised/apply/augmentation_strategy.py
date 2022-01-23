'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from self_supervised.apply import config

ft_exp_augmentation = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.3),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])