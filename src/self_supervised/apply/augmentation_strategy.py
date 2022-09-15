from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from self_supervised.apply import config

pretrain_augmentation = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.2)
        ])

pretrain_v2_augmentation = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        A.Flip(p=0.3),
        A.Rotate(p=0.3)
        #A.Affine(translate_percent = 0.05, p=0.2)
        ])

