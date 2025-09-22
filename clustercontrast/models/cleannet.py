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
       
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x), y
    

def prep_img(xc):
    
    [B, W, H] = xc.size()
    xc = xc.unsqueeze(1) 
    xc=xc.repeat(1, 3,1,1)

    return xc         

class UNet(nn.Module):
    def __init__(self, depth=0, pretrained=True, cut_at_pooling=False, img_size = 50, 
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
        
        prebase = nn.Sequential(model.encoder1, model.pool1, model.encoder2, model.pool2, model.encoder3, model.pool3,
                                  model.encoder4, model.pool4) # , model.bottleneck)

        self.bases = nn.ModuleList([
            nn.Sequential(prebase, model.bottleneck) for _ in range(new_in_channels)
        ])
        # freeze the pretrained model
        # for base in self.bases:
        #     for param in base.parameters():
        #         param.requires_grad = False
        self.eca = eca_layer(new_in_channels)
        self.SpatialGate = SpatialGate()  
        out_planes =512 *  val*val # val = img_size // 16 = 50/16 = 3 
  
        if not self.cut_at_pooling: #True
  
            self.norm = norm 
            self.dropout = dropout
            self.has_embedding = num_features > 0 #False
            self.num_classes = num_classes
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

        processed_channels = [base(prep_img(x[:, i, :, :])) for i, base in enumerate(self.bases)]
        x = torch.cat(processed_channels, dim=1)
       
        # ATTENTION---------------------------------------------------------------------------------------
        x, attention_weights = self.eca(x) 
        x = self.SpatialGate(x) 
        x= torch.reshape(x,(B, C, -1)) #reshape to be like the unet 1
        x=torch.sum(x, axis = 1 ) #sum along channel? #otherwise it's B * 512*C * w*h , no sum? 
        bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x, attention_weights #final output when testing

        if self.norm: #False
            bn_x = F.normalize(bn_x)
        elif self.has_embedding: #False
            bn_x = F.relu(bn_x)

        if self.dropout > 0: 
            bn_x = self.drop(bn_x)

        if self.num_classes > 0: 
            prob = self.classifier(bn_x)
        else:
            return bn_x, attention_weights #final output when training

        return prob, attention_weights


def unet(**kwargs):
    return UNet(**kwargs)
