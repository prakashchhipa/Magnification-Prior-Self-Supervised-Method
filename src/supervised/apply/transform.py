'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

from albumentations.core.composition import Transforms
from cv2 import transform
from torchvision import transforms as t
import albumentations as A
from albumentations.pytorch import ToTensorV2

from supervised.apply.auto_augment import AugMix

from self_supervised.apply import config

resize_transform = t.Compose([
        t.ToPILImage(), 
        t.Resize((341, 341)),
        t.ToTensor()
        ])

resize_transform_bach_512 = t.Compose([
        t.ToPILImage(), 
        t.Resize((512, 512)),
        t.ToTensor()
        ])

resize_transform_bach_512_augmix = t.Compose([
        t.ToPILImage(), 
        t.Resize((512, 512)),
        AugMix(),
        t.ToTensor()
        ])

resize_transform_bach_224_augmix = t.Compose([
        t.ToPILImage(), 
        t.Resize((224, 224)),
        AugMix(),
        t.ToTensor()
        ])

resize_transform_bach_224 = t.Compose([
        t.ToPILImage(),
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

resize_rcfor1_transform = t.Compose([
        t.ToPILImage(), 
        t.Resize((224, 224)),
        t.ToTensor()
        ])

resize_rcfor1_transform_v2 = t.Compose([
        t.ToPILImage(), 
        t.Resize((230, 350)),
        t.ToTensor()
        ])

train_transform = t.Compose([
        t.ToPILImage(), 
        t.ToTensor()
        ])
