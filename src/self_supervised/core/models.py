'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


'''EfficientNet pretrained with MLP head for contrastive self-supervised training'''      
class EfficientNet_MLP(torch.nn.Module):
    def __init__(self, features_dim=2048, v='b2', mlp_dim=2048):
        super(EfficientNet_MLP, self).__init__()
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{v}')
        backbone_dim = self.backbone._fc.in_features
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()
        self.backbone._swish = nn.Identity()
        self.mlp = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(backbone_dim, mlp_dim),
                    nn.BatchNorm1d(mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.BatchNorm1d(mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.BatchNorm1d(mlp_dim),
        )
        self.head = nn.Sequential(
                    
                    nn.Linear(mlp_dim, mlp_dim//4),
                    nn.BatchNorm1d(mlp_dim//4),
                    nn.ReLU(),
                    nn.Linear(mlp_dim//4, features_dim),
        )
    def backbone_mlp(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        return x
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        x = self.head(x)
        x = F.normalize(x, dim = -1)
        return x