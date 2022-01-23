'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

from albumentations.core.composition import Transforms
from cv2 import transform
from torchvision import transforms as t
import albumentations as A
from albumentations.pytorch import ToTensorV2

from self_supervised.apply import config

# Dataset input processing - trainset
train_transform = t.Compose([
        t.ToPILImage(), 
        t.ToTensor()
        ])
