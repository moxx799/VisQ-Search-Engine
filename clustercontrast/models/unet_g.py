import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .pooling import build_pooling_layer
from .GLAM import * 

    
class UNet(nn.Module):
    def __init__(self, depth=0, pretrained=True, cut_at_pooling=False, img_size = 175, 
                num_features=0, new_in_channels=7, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(UNet, self).__init__()
        val = img_size
        for i in range(4): 
            nval = int(val  / 2)
            val = nval
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        #Load the pretrained UNet
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                            in_channels=3, out_channels=1, init_features=32, pretrained=self.pretrained)

        #Extract just the encoder half
        prebase = nn.Sequential(model.encoder1, model.pool1, model.encoder2, model.pool2, model.encoder3, model.pool3,
                                  model.encoder4, model.pool4) # , model.bottleneck)
        for param in prebase.parameters():
           param.requires_grad = False #freeze the weights of the encoder except for the bottleneck 
        
        self.base = nn.Sequential(prebase, model.bottleneck)

        print(self.base)
        #Add pooling
        self.gap = build_pooling_layer(pooling_type)
        self.AM = GLAM(in_channels=new_in_channels, num_reduced_channels=new_in_channels, feature_map_size=512, kernel_size=1)
        # self.multihead_attn = nn.MultiheadAttention(new_in_channels, 5)
        
        out_planes1 =  512 *  val*val # 51200 # 512 *  9#new_in_channels #2560
        out_planes = 2500 #512 *  9 #2500 
        self.FC = nn.Sequential(nn.Linear(out_planes1, out_planes1), nn.Linear(out_planes1, out_planes), nn.Linear(out_planes, out_planes)) 


        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            # out_planes =  512 *  9 #new_in_channels #2560

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else: #Add batch normalization really is all for default
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

        x = self.base(x) #Unet's encoder, B*C x 512
        # print('Original size after encoding: ', x.size() )
        [b, v, w, h ] = x.size()
        # x = torch.reshape(x, (B, C, -1) ) #expand back to the channel size
        #reshape for per channel and then channel attention
        #x size B X C X vector size
        x= torch.reshape(x,(B, C, v, -1)) #reshape to just batch x channel x 512
        # print('Reshaped tensor after unet encoder: ', x.size() )
        # attention_x=self.AM(x) #apply attention
        # print('shape after attention: ', attention_x.size() ) # batch x channel x v x 81
        # x = self.MA(x, attention_x)
        # attn_output, attn_output_weights = multihead_attn(x, key, value)

        # ATTENTION---------------------------------------------------------------------------------------
        # x=self.ChannelGate(x)
        # x = self.eca(x) 
        # x = self.SpatialGate(x) #adding the spatial gate 
        x=self.AM(x) 

        # print('Attention shape: ', x.size() )

        x=torch.sum(x, axis = 1 ) #sum along channel?
        # print('Sum size: ', x.size() ) # 50x512x100

        x= torch.reshape(x,(B, -1)) #reshape to just batch
        # print('Sum size: ', x.size() ) # 50x512x100


        x=self.FC(x)
        
        # print('Reshape size: ', x.size() )
        # x = self.gap(x)
        # # x = x.view(x.size(0), -1)
        # print(x.size() )

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





def unet(**kwargs):
    return UNet(**kwargs)
