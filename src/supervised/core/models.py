'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

from email.policy import strict
import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision
from efficientnet_pytorch import EfficientNet

class ResNet_Model(torch.nn.Module):
    def __init__(self, version, pretrained=False, num_classes = 1, use_sigmoid_head = True):
        super(ResNet_Model, self).__init__()
        self.num_classes = num_classes
        self.model = None
        self.use_sigmoid_head = use_sigmoid_head
        self.num_ftrs = 0
        if 18 == version:
            self.model = models.resnet18(pretrained=pretrained)
        elif 50 == version:
            self.model = models.resnet50(pretrained=pretrained)
        elif 101 == version:
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise error("Select ResNet architecture from 18 | 50 | 101")
        self.num_ftrs=self.model.fc.in_features
        
        self.model.fc = nn.Sequential(nn.Dropout(0.8), nn.Linear(self.num_ftrs, 1))#(nn.Linear(self.num_ftrs, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output = None
        if True == self.use_sigmoid_head:
            output = self.sig(self.model(x))
        else:
            output = self.model(x)
        return output


class ResNet(torch.nn.Module):
    def __init__(self, version, pretrained=True, num_classes = 1):
        super(ResNet, self).__init__()
        #new setting
        self.num_classes = num_classes
        self.model = getattr(models, f'resnet{version}')(pretrained=pretrained)
        self.num_ftrs=self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4), 
            nn.Linear(self.num_ftrs, 1)
            )
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        output = self.sig(self.model(x))
        return output


class EfficientNet_Model(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet_Model, self).__init__()
        num_classes = 2
        self.model = EfficientNet.from_pretrained("efficientnet-b2")
        num_ftrs=self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        output = self.sig(self.model(x))
        return output

    
BatchNorm = nn.BatchNorm2d

# More flexible Dilated ResNet (Any ResNet architecture)

class ResNetDilated(nn.Module):
    """ ResNet backbone with dilated convolutions """
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResNetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward_stage(self, x, stage):
        assert(stage in ['conv','layer1','layer2','layer3','layer4', 'layer1_without_conv'])
        
        if stage == 'conv':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x

        elif stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x
        
        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer(x)

    def forward_stage_except_last_block(self, x, stage):
        assert(stage in ['layer1','layer2','layer3','layer4'])

        if stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1[:-1](x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer[:-1](x)
    
    def forward_stage_last_block(self, x, stage):
        assert(stage in ['layer1','layer2','layer3','layer4'])

        if stage == 'layer1':
            x = self.layer1[-1](x)
            return x

        else: # Stage 2, 3 or 4
            layer = getattr(self, stage)
            return layer[-1](x)

    def get_last_shared_layer(self):
        return self.layer4
    
class ResNetDilated_BACH(nn.Module):
    def __init__(self, version=50, dilate_scale=8, num_classes = 4):
        super(ResNetDilated_BACH, self).__init__()
        if 50 == version:
            orig_resnet = torchvision.models.resnet50(pretrained=True)
        elif 101 == version:
            orig_resnet = torchvision.models.resnet101(pretrained=True)
        else:
            raise error("version not implemented yet")
        #dilated ResNet model
        self.model = ResNetDilated(orig_resnet=orig_resnet, dilate_scale=dilate_scale)
        self.num_classes = num_classes
        #self.num_ftrs=self.model.fc.in_features
        self.fc = nn.Linear(2048, self.num_classes)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1))
    def forward(self, x):
        x = self.model(x)
        x=self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
        
        
        
        