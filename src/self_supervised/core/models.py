'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

from email.policy import strict
import torchvision.models as models
import torch.nn as nn
from torchvision.models.resnet import Bottleneck,_resnet
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Resnet_SSL(torch.nn.Module):
    def __init__(self, version=50, projector=None, supervised_pretrained=None, simclr_pretrained=None):
        super(Resnet_SSL, self).__init__()
        if 18 == version:
            self.backbone = models.resnet18(pretrained = supervised_pretrained)
        elif 34 == version:
            self.backbone = models.resnet34(pretrained = supervised_pretrained)
        elif 50 == version:
            self.backbone = models.resnet50(pretrained = supervised_pretrained)
        elif 101 == version:
            self.backbone = models.resnet101(pretrained = supervised_pretrained)
        elif 152 == version:
            self.backbone = models.resnet152(pretrained = supervised_pretrained)

        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.backbone(x))
        # z2 = self.projector(self.backbone(x2))
        return F.normalize(self.bn(z), dim = -1)

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


class Dilated_Resnet_SSL(torch.nn.Module):
    def __init__(self, version=50, projector=None, supervised_pretrained=None, simclr_pretrained=None, dilate_scale=8):
        super(Dilated_Resnet_SSL, self).__init__()
        if 18 == version:
            orig_resnet = models.resnet18(pretrained = supervised_pretrained)
        elif 34 == version:
            orig_resnet = models.resnet34(pretrained = supervised_pretrained)
        elif 50 == version:
            orig_resnet = models.resnet50(pretrained = supervised_pretrained)
        elif 101 == version:
            orig_resnet = models.resnet101(pretrained = supervised_pretrained)
        elif 152 == version:
            orig_resnet = models.resnet152(pretrained = supervised_pretrained)
        else:
            raise ValueError("input version for ResNet is either not supported or not correct")

        self.backbone = ResNetDilated(orig_resnet=orig_resnet, dilate_scale=dilate_scale)
        self.adaptivepool = torch.nn.AdaptiveAvgPool2d((1))
        #self.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptivepool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        z = self.projector(x)
        # z2 = self.projector(self.backbone(x2))
        return F.normalize(self.bn(z), dim = -1)


class EfficientNet_SSL(torch.nn.Module):
    def __init__(self, version='b2', projector=None, supervised_pretrained=None, simclr_pretrained=None):
        super(EfficientNet_SSL, self).__init__()
        if True == simclr_pretrained:
            raise ValueError("ImageNet SimCLR pretrained weights not available for EfficientNet as of now")
        if True == supervised_pretrained:
            self.backbone = EfficientNet.from_pretrained(f'efficientnet-{version}')
        else:
            self.backbone = EfficientNet.from_name(f'efficientnet-{version}')
        
        backbone_dim = self.backbone._fc.in_features
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()
        self.backbone._swish = nn.Identity()
        
        # projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
    
    def forward(self, x):
        x = self.backbone(x)
        z = self.projector(x)
        return F.normalize(self.bn(z), dim = -1)


'''ResNet101pretrained with MLP head for contrastive self-supervised training'''    
class Resnet101_MLP(torch.nn.Module):
    def __init__(self, features_dim=128, mlp_dim=2048, version='50', device= None, supervised_pretrained= None, simclr_pretrained=None):
        
        super(Resnet101_MLP, self).__init__()
        self.backbone = None
        
        if True == supervised_pretrained:
            print('START - Resnet101_MLP supervised imageNet pretrained encoder weights loading')
            self.backbone = models.resnet101(pretrained=True)
            print('STOP - Resnet101_MLP supervised imageNet pretrained encoder weights loading')
        else:
            print('START - Resnet101_MLP without supervised imageNet weights')
            self.backbone = models.resnet101(pretrained=False)
            print('START - Resnet101_MLP without supervised imageNet weights')
        if True == simclr_pretrained:
            print('START - Resnet101_MLP simCLR imageNet-1k pretrained encoder weights loading - VISSL')
            self.backbone = models.resnet101()
            self.backbone.load_state_dict(torch.load('/home/prachh/pretrained/converted_vissl_resnet101.torch'), strict = False)
            print('STOP - Resnet101_MLP simCLR imageNet-1k pretrained encoder weights loading - VISSL')
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
    
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
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.BatchNorm1d(mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, features_dim),
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


'''ResNet50 pretrained with MLP head for contrastive self-supervised training'''    
class Resnet50_MLP(torch.nn.Module):
    def __init__(self, features_dim=2048, mlp_dim=2048, version='50', device= None, supervised_pretrained=None, simclr_pretrained = None):
        
        super(Resnet50_MLP, self).__init__()
        self.backbone = None
        
        if True == supervised_pretrained:
            print('START - Resnet50_MLP supervised imageNet pretrained encoder weights loading')
            self.backbone = models.resnet50(pretrained=True)
            print('STOP - Resnet50_MLP supervised imageNet pretrained encoder weights loading')
        else:
            print('START - Resnet50_MLP without supervised imageNet weights')
            self.backbone = models.resnet50(pretrained=False)
            print('START - Resnet50_MLP without supervised imageNet weights')
        if True == simclr_pretrained:
            print('START - Resnet50_MLP simCLR imageNet-1k pretrained encoder weights loading - VISSL')
            self.backbone = models.resnet50()
            self.backbone.load_state_dict(torch.load('/home/prachh/pretrained/converted_vissl_resnet50.torch'), strict = False)
            print('STOP - Resnet50_MLP simCLR imageNet-1k pretrained encoder weights loading - VISSL')
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
    
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
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.BatchNorm1d(mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, features_dim),
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