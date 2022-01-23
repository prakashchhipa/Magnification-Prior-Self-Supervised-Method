'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

from albumentations.core.composition import Transforms
from cv2 import transform
from torchvision import transforms as t
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.transforms import RandomCrop, Resize

from self_supervised.apply import config

# Dataset input processing - trainset

resize_transform = t.Compose([
        t.ToPILImage(), 
        t.Resize((341, 341)),
        #t.Resize((224,224)),
        #A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        t.ToTensor()
        ])