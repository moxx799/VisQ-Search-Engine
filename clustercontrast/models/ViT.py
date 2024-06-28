import torch 
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .pooling import build_pooling_layer


class ViT(nn.Module): 
    def __init__(self, depth=0, pretrained=True, cut_at_pooling=False,
                num_features=0, new_in_channels=7, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(ViT, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        #Load the pretrained UNet
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        # print(model)
        # self.base = nn.Sequential(model.encoder1, model.pool1, model.encoder2, model.pool2, model.encoder3, model.pool3, 
                                #   model.encoder4, model.pool4, model.bottleneck)
        self.base = model # nn.Sequential(*list(model.children())[:-2])
        # print(self.base)
        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = 1000 * new_in_channels #2560

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


    def forward(self, x):
        bs = x.size(0)
        [B, C, W, H] = x.size() 
        x= torch.reshape(x, (B*C, -1, W, H) ) #convert to grayscale
        x=x.expand(-1, 3, -1, -1) #make rgb 
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False) #enlarge for encoder

        x = self.base(x)
        # print(x.size())
        # x = self.gap(x)
        # print(x.size())
        # x = x.view(x.size(0), -1)
        x= torch.reshape(x,(B, -1)) #reshape to just batch
        # print(x.size() )

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





def vit(**kwargs):
    return ViT(**kwargs)
