import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .pooling import build_pooling_layer

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,groups =1, dilation=1,bias=False):
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
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
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
        # print(channel_att_sum.size() )
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # scale = F.sigmoid( channel_att_sum ).expand_as(x)
        # print('Sizes: ', scale.size(), x.size() )
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv = SeparableConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
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

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x), y
    
# class multi_base(nn.Module): 
#     def __init__(self, new_in_channels, prebase):
#         super(multi_base, self).__init__()
#         self.enc = nn.ModuleList() 
#         for i in range(new_in_channels): 
#             self.enc.append(prebase)
#     def forward(self,x): 
#         x= torch.reshape(x, (B*C, -1, W, H) ) #convert to grayscale
#         x_vecs=[] 
#         for i in x: 
#             layer = self.enc[i]

def prep_img(xc):
    [B, W, H] = xc.size()
    # xc= torch.reshape(xc, (B, None, W, H) ) #convert to grayscale
    xc.unsqueeze_(1)
    # print(xc.size() )
    xc=xc.repeat(1, 3,1,1)
    # print('grey: ', xc.size() )
    # xc=xc.expand(-1, 3, -1, -1) #make rgb
    return xc         

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
        #for param in prebase.parameters():
        #    param.requires_grad = False #freeze the weights of the encoder except for the bottleneck 
        
        if new_in_channels == 4: 
            self.base1 = nn.Sequential(prebase, model.bottleneck)
            self.base2 = nn.Sequential(prebase, model.bottleneck)
            self.base3 = nn.Sequential(prebase, model.bottleneck)
            self.base4 = nn.Sequential(prebase, model.bottleneck)
        elif new_in_channels == 5: 
            self.base1 = nn.Sequential(prebase, model.bottleneck)
            self.base2 = nn.Sequential(prebase, model.bottleneck)
            self.base3 = nn.Sequential(prebase, model.bottleneck)
            self.base4 = nn.Sequential(prebase, model.bottleneck)
            self.base5 = nn.Sequential(prebase, model.bottleneck)
        elif new_in_channels == 8: 
            self.base1 = nn.Sequential(prebase, model.bottleneck)
            self.base2 = nn.Sequential(prebase, model.bottleneck)
            self.base3 = nn.Sequential(prebase, model.bottleneck)
            self.base4 = nn.Sequential(prebase, model.bottleneck)
            self.base5 = nn.Sequential(prebase, model.bottleneck)
            self.base6 = nn.Sequential(prebase, model.bottleneck)
            self.base7 = nn.Sequential(prebase, model.bottleneck)
            self.base8 = nn.Sequential(prebase, model.bottleneck)

        # print(self.base)
        #Add pooling
        # self.gap = build_pooling_layer(pooling_type)
        # self.AM = attention_model(in_chans=new_in_channels, out_chans=new_in_channels)
        # self.MA = Multiply()
        self.eca = eca_layer(new_in_channels)
        # self.ChannelGate = ChannelGate(new_in_channels*512, reduction_ratio=4)
        self.SpatialGate = SpatialGate()
        # self.multihead_attn = nn.MultiheadAttention(new_in_channels, 5)
        
        out_planes1 = 512 *  val*val # val*val*new_in_channels # 51200 # 512 *  9#new_in_channels #2560
        out_planes = out_planes1 # 2500 #512 *  9 #2500 
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
        # x= torch.reshape(x, (B*C, -1, W, H) ) #convert to grayscale
        # x=x.expand(-1, 3, -1, -1) #make rgb
        # print(prep_img(x[:,0,:,:]).size())
        if C == 4: 
            x1 = self.base1(prep_img(x[:,0,:,:]))
            x2 = self.base2(prep_img(x[:,1,:,:]))
            x3 = self.base3(prep_img(x[:,2,:,:]))
            x4 = self.base4(prep_img(x[:,3,:,:]))
            x = torch.cat([x1,x2,x3,x4], dim=1)
        elif C ==5: 
            x1 = self.base1(prep_img(x[:,0,:,:]))
            x2 = self.base2(prep_img(x[:,1,:,:]))
            x3 = self.base3(prep_img(x[:,2,:,:]))
            x4 = self.base4(prep_img(x[:,3,:,:]))
            x5 = self.base5(prep_img(x[:,4,:,:]))
            x = torch.cat([x1,x2,x3,x4, x5], dim=1)
        elif C ==8: 
            x1 = self.base1(prep_img(x[:,0,:,:]))
            x2 = self.base2(prep_img(x[:,1,:,:]))
            x3 = self.base3(prep_img(x[:,2,:,:]))
            x4 = self.base4(prep_img(x[:,3,:,:]))
            x5 = self.base5(prep_img(x[:,4,:,:]))
            x6 = self.base3(prep_img(x[:,5,:,:]))
            x7 = self.base4(prep_img(x[:,6,:,:]))
            x8 = self.base5(prep_img(x[:,7,:,:]))
            x = torch.cat([x1,x2,x3,x4, x5, x6, x7, x8], dim=1)
        # x = self.base(x) #Unet's encoder, B*C x 512
        # print('Original size after encoding: ', x.size() ) 
        #batch, channels, dim, w, h 
        # [b, v, w, h] = x.size() 
        # [b, v, w, h ] = x.size()
        # x = torch.reshape(x, (B, C, -1) ) #expand back to the channel size
        #reshape for per channel and then channel attention
        #x size B X C X vector size
        # x= torch.reshape(x,(B, C, v, -1)) #reshape to just batch x channel x 512 << uncomment????
        # print('Reshaped tensor after unet encoder: ', x.size() )
        # attention_x=self.AM(x) #apply attention
        # print('shape after attention: ', attention_x.size() ) # batch x channel x v x 81
        # x = self.MA(x, attention_x)
        # attn_output, attn_output_weights = multihead_attn(x, key, value)

        # ATTENTION---------------------------------------------------------------------------------------
        # x=self.ChannelGate(x)
        # [b, v, w, h ] = x.size()
        # x= torch.reshape(x,(B, C, -1 , w*h))
        #do the attention on the tiny patch for all the features 
        x, attention_weights = self.eca(x) 
        x = self.SpatialGate(x) #adding the spatial gate

        # x = self.SpatialGate(x) #adding the spatial gate 

        # print('Attention shape: ', x.size() )

        # x=torch.sum(x, axis = 1 ) #sum along channel?
        # print('Sum size: ', x.size() ) # 50x512x100,  batch x3x3

        x= torch.reshape(x,(B, C, -1)) #reshape to be like the unet 1

        
        # print('Spatial Gate: ', x.size() ) # 50x512x100
        # x= torch.reshape(x,(B, -1)) #reshape to just batch
        x=torch.sum(x, axis = 1 ) #sum along channel? #otherwise it's B * 512*C * w*h , no sum? 
        # print('Sum size: ', x.size() ) # 50x512x100,  batch x3x3

        # x= torch.reshape(x,(B, -1)) #reshape to just batch B*C*W*H

        #x=self.FC(x)
        
        # print('Reshape size: ', x.size() )
        # x = self.gap(x)
        # # x = x.view(x.size(0), -1)
        # print(x.size() )

        # print(x.size() )
        if self.cut_at_pooling:
            return x, attention_weights

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x, attention_weights

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x, attention_weights

        return prob, attention_weights





def unet(**kwargs):
    return UNet(**kwargs)
