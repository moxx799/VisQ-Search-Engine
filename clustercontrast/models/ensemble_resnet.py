from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer

from . import *

#------------------------Ensemble Model-----------------------

class EnsembleModel(nn.Module):   

    def __init__(self, args):
        super().__init__()

        #create three identical models 
        self.modelA = models.create(args.arch, num_features=args.features, new_in_channels=args.num_channels, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
        self.modelB = models.create(args.arch, num_features=args.features, new_in_channels=args.num_channels, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
        self.modelC = models.create(args.arch, num_features=args.features, new_in_channels=args.num_channels, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
        
        out_planes = self.modelA.fc.in_features #define the output size 

        #fully connected layer to concatenate the three models 
        self.FC = nn.Linear(out_planes * 3, out_planes)
        
    def forward(self, x):
        #all models are the same? 
        #all inputs are at different scales 
        x1 = self.modelA(x[0]) #scale 1
        x2 = self.modelB(x[1]) #scale 2
        x3 = self.modelC(x[2]) #scale 3
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.FC(x)
        return out