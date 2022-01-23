'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNet_Model(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet_Model, self).__init__()
        num_classes = 2
        self.model = EfficientNet.from_pretrained("efficientnet-b2")
        num_ftrs=self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
        self.sig = nn.Sigmoid()
        #self.model.fc=nn.Linear(512,num_classes)
    def forward(self, x):
        output = self.sig(self.model(x))
        return output