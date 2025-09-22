import torch
from torch import nn
from torch.nn import functional as F
from transformers import BeitModel, BeitConfig
from torch.nn import init

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # Shape: [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # Shape: [B, C, 1] -> [B, 1, C]
        y = self.conv(y).contiguous()  # Ensure contiguity after convolution
        y = y.transpose(-1, -2).unsqueeze(-1)  # Shape: [B, C, 1] -> [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x), y

class BEiTv3Encoder(nn.Module):
    def __init__(self, depth=0, pretrained=True, cut_at_pooling=False, img_size=224,
                 num_features=0, new_in_channels=7, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(BEiTv3Encoder, self).__init__()
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.new_in_channels = new_in_channels
        self.img_size = img_size
        self.num_features = num_features

        # Load BEiTv3 configuration and model
        config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        config.image_size = img_size
        config.output_attentions = False  # Disable attention outputs
        config.output_hidden_states = False  # Disable extra hidden states
        self.beit = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k", config=config)
        self.frozen_layers = 10
        # Freeze the first 9 layers
        for i, layer in enumerate(self.beit.encoder.layer):
            if i < 11:  # Freeze first 9 layers
                for param in layer.parameters():
                    param.requires_grad = False
            else:  # Keep last 3 layers trainable
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Also freeze the embeddings
        for param in self.beit.embeddings.parameters():
            param.requires_grad = False

        # Feature dimension adjustments
        self.feature_dim = 768  # BEiT base hidden size
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.output_size = 3  # Target spatial size

        # Adaptive pooling and projection to match UNet's output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        self.channel_reduction = nn.Conv2d(self.feature_dim, 512, kernel_size=1)

        # Attention mechanisms
        self.eca = eca_layer(new_in_channels)
        self.spatial_gate = SpatialGate()

        # Calculate final feature dimension
        self.final_feat_dim = 512 * self.output_size * self.output_size

        # Classifier and BN layers
        self.norm = norm
        self.dropout = dropout
        self.num_classes = num_classes
        self.feat_bn = nn.BatchNorm1d(self.final_feat_dim)
        self.feat_bn.bias.requires_grad_(False)

        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        if num_classes > 0:
            self.classifier = nn.Linear(self.final_feat_dim, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

    def duplicate_channel(self, x):
        """Duplicate single channel to 3 channels for BEiT input"""
        return x.repeat(1, 3, 1, 1).contiguous()
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.contiguous()
        processed_features = []

        for i in range(C):
            x_channel = x[:, i:i+1, :, :]  # Shape: [B, 1, H, W]
            
            # Duplicate single channel to 3 channels for BEiT input
            x_channel_3c = self.duplicate_channel(x_channel)
            
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                last_hidden_state = self.beit(x_channel_3c).last_hidden_state

            # print('shape of outputs.last_hidden_state:', last_hidden_state.shape)
            # Reshape to spatial format
            patch_dim = int(self.num_patches ** 0.5)
            # inside BEiTv3Encoder.forward, right before calling self.beit(...)
            features = last_hidden_state[:, 1:].reshape(B, patch_dim, patch_dim, -1).permute(0, 3, 1, 2).contiguous()  # Shape: [B, 768, patch_dim, patch_dim]
            # print('shape of features1:', features.shape)
            # Adaptive pooling and channel reduction
            features = self.adaptive_pool(features)
            # print('shape of features2:', features.shape)
            features = self.channel_reduction(features)  # Shape: [B, 512, 3, 3]
            # print('shape of features3:', features.shape)
            processed_features.append(features)
            
        x = torch.cat(processed_features, dim=1).contiguous()
        x, attn_weights = self.eca(x)
        x = x.reshape(B, C, -1)
        x = torch.sum(x, dim=1)
        bn_x = self.feat_bn(x)
        
        if not self.training:
            bn_x = F.normalize(bn_x)
            return bn_x, attn_weights
        
        if self.norm:
            bn_x = F.normalize(bn_x)
        
        if self.dropout > 0:
            bn_x = self.drop(bn_x)
        
        return bn_x, attn_weights

    def _debug_forward(self, x):
        B, C, H, W = x.shape
        processed_features = []
        x = x.contiguous()
        # sum x into 3 channels if C > 3

        # Process each channel separately
        for i in range(C):
            x_channel = x[:, i:i+1, :, :]  # Shape: [B, 1, H, W]
            
            # Duplicate single channel to 3 channels for BEiT input
            x_channel_3c = self.duplicate_channel(x_channel)
            
            last_hidden_state = self.beit(x_channel_3c).last_hidden_state #[B 197, 768]

            # print('shape of outputs.last_hidden_state:', last_hidden_state.shape)
            # Reshape to spatial format
            patch_dim = int(self.num_patches ** 0.5)
            features = last_hidden_state[:, 1:].reshape(B, patch_dim, patch_dim, -1).permute(0, 3, 1, 2).contiguous()  # Shape: [B, 768, patch_dim, patch_dim]
            # print('shape of features1:', features.shape)
            # Adaptive pooling and channel reduction
            features = self.adaptive_pool(features)
            # print('shape of features2:', features.shape)
            features = self.channel_reduction(features)  # Shape: [B, 512, 3, 3]
            # print('shape of features3:', features.shape)
            processed_features.append(features)
        
        # Concatenate along channel dimension
        x = torch.cat(processed_features, dim=1)  # Shape: [B, C*512, 3, 3]
        # print('shape of x after concat:', x.shape)
        # # Apply attention mechanisms
        x, attn_weights = self.eca(x)
      
        # print('shape of x after eca:', x.shape)
        x = self.spatial_gate(x)
        # print('shape of x after spatial gate:', x.shape)
        # Flatten and sum over channels
        x = x.reshape(B, C, -1)  # Shape: [B, C, 512*3*3] [B,C,4608]
        # print('shape of x after reshape:', x.shape)
        x = torch.sum(x, dim=1)   # Shape: [B, 512*3*3] [B,4608]
        # print('shape of x after sum:', x.shape)
        # Apply batch norm
        print('shape of x before bn:', x.shape) # [B, 512, 3, 3]
        x = x.view(B, -1)
        bn_x = self.feat_bn(x)
        
        # print('shape of bn_x:', bn_x.shape)

        
        if not self.training:
            bn_x = F.normalize(bn_x)
            
            return bn_x, attn_weights
        
        if self.norm:
            bn_x = F.normalize(bn_x)
        
        if self.dropout > 0:
            bn_x = self.drop(bn_x)
    
        
        return bn_x, attn_weights

def beit_encoder(**kwargs):
    return BEiTv3Encoder(**kwargs)