import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .pooling import build_pooling_layer

class SeparableConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x)
        print('Sizes: ', scale.size(), x.size() )
        return x * scale

class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()
    self.SM = nn.Softmax()
  def forward(self, x, attentionX ):
    x = self.SM(x)
    # print('original shape vs attention shape: ', x.size(), attentionX.size() )
    res = torch.multiply(x, attentionX)
    return res

class attention_model(nn.Module):
    def __init__(self,in_chans, out_chans):
        super(attention_model,self).__init__()
        self.conv1 = nn.Conv2d(in_chans,in_chans,5,1,padding=5//2,bias=False)
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.mp1 = nn.MaxPool2d(1)
        self.relu1 = nn.ReLU()
        self.drop1= nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_chans,out_chans,5, 1, padding=5//2, bias=False)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.mp2 = nn.MaxPool2d(1)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        B, C = x.size(0), x.size(1)
        x= self.conv1(x)
        x = self.bn1(x)
        x= self.mp1(x)
        x= self.relu1(x)
        x=self.drop1(x)
        # print('After one conv pass:', x.size() )
        x= self.conv2(x)
        x= self.bn2(x)
        x=self.mp2(x)
        x=self.relu2(x)
        return x


class UNet(nn.Module):
    def __init__(self, depth=0, pretrained=True, cut_at_pooling=False,
                num_features=0, new_in_channels=7, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(UNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        #Load the pretrained UNet
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                            in_channels=3, out_channels=1, init_features=32, pretrained=self.pretrained)
        #Extract just the encoder half
        self.base = nn.Sequential(model.encoder1, model.pool1, model.encoder2, model.pool2, model.encoder3, model.pool3,
                                  model.encoder4, model.pool4, model.bottleneck)
        #Add pooling
        self.gap = build_pooling_layer(pooling_type)
        self.AM = attention_model(in_chans=new_in_channels, out_chans=new_in_channels)
        self.MA = Multiply()
        self.ChannelGate = ChannelGate(new_in_channels)
        # self.multihead_attn = nn.MultiheadAttention(new_in_channels, 5)



        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = 512 * new_in_channels #2560

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

        '''
        def attention_model(xx,ch):
            w = Conv1D(512, kernel_size=(5), strides=(1,1), padding='same', use_bias=False)(xx)
            w = BatchNormalization()(w)
            w = MaxPooling2D((1, 1), padding='same')(w)
            w = LeakyReLU()(w)
            w = Dropout(0.2)(w)

            w = Conv1D(ch, kernel_size=(1), strides=(1,1), padding='same', use_bias=False)(w)
            w = BatchNormalization()(w)
            w = MaxPooling2D((1, 1), padding='same')(w)
            w = LeakyReLU()(w)
            print('w.shape: ', w.shape)
            return w

        def something(x,w):
            x = PixelSoftmax(axis=-1)(x)
            print('x0shape',x.shape)
            x = Reshape((ps,ps, reg))(x)
            x = Multiply()([tf.expand_dims(w,3),x])
            return(x)'''

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
        x=self.ChannelGate(x)
        print('Final shape: ', x.size() )

        x = self.gap(x)
        # # x = x.view(x.size(0), -1)
        print(x.size() )
        x= torch.reshape(x,(B, -1)) #reshape to just batch
        print(x.size() )
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
