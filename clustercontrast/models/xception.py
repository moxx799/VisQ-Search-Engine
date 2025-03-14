from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer
from .xception_model import *


__all__ = ['Xception', 'xception']


class Xception(nn.Module):
    __factory = {
            1: xception(1000), 
    } 
    def __init__(self, depth, pretrained=True, cut_at_pooling=False, num_features=0, new_in_channels=7, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(Xception, self).__init__()

        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        

        # xception= xception(num_classes=num_classes, pretrained='imagenet')
        xception = Xception.__factory[depth]# (pretrained='imagenet')
        #----------RWM add to more channels ------------------------------------
        # weight = xception.conv1.weight.clone() #copy over weight to clone RWM
        # xception.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
        # print(xception)
        # new_in_channels = 7
        layer = xception.conv1
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=new_in_channels,
                          out_channels=layer.out_channels,
                          kernel_size=layer.kernel_size,
                          stride=layer.stride,
                          padding=layer.padding,
                          bias=layer.bias)
        copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights
        # Copying the weights from the old to the new layer
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()
            #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            for i in range(new_in_channels - layer.in_channels):
                channel = layer.in_channels + i
                new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)
        xception.conv1 = new_layer
        # print(xception)
        self.base = torch.nn.Sequential(*(list(xception.children())[:-1]))
        #---------------------------------------------------------


        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = xception.last_linear.in_features#xception.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        bs = x.size(0)

        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def xception(**kwargs):
    return Xception(1, **kwargs)

