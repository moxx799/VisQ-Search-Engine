import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class ViTBasedModel(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False, img_size=175,
                 num_features=0, new_in_channels=7, norm=False, dropout=0,
                 num_classes=0, pooling_type='avg'):
        super(ViTBasedModel, self).__init__()
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.new_in_channels = new_in_channels
        self.hidden_dim = 1024 #768  # DeiT-base hidden dimension
        # Load pretrained DeiT
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self._freeze_vit_layers(0.5)
        # Modify forward to get all tokens
        # self.original_forward = self.vit.forward
        # self.vit.forward = self.vit_forward
        
        # Attention modules
        self.eca = ECALayer(channel=new_in_channels * self.hidden_dim)
        self.spatial_gate = SpatialGate()

        # Feature processing
        out_planes = new_in_channels * self.hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(out_planes, out_planes // 2),
            nn.ReLU(),
            nn.Linear(out_planes // 2, out_planes // 4)
        )

        # Embedding and classification
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
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
        
    def replicate_channel_to_rgb(self, channel_tensor):
        """Convert single-channel to normalized RGB"""
        x = F.interpolate(channel_tensor, size=224, mode='bilinear')
        x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std
    
    def _freeze_vit_layers(self, frac=0.7):
        """Freeze fraction of ViT layers for transfer learning"""
        num_blocks = len(self.vit.blocks)
        freeze_until = int(num_blocks * frac)
        for i, block in enumerate(self.vit.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
     
    # def vit_forward(self, x):
    #     device = x.device  # Ensure everything runs on the same device
    #     B = x.shape[0]
        
    #     # Ensure tensors are moved to the same device
    #     self.vit.patch_embed = self.vit.patch_embed.to(device)
    #     self.vit.cls_token = self.vit.cls_token.to(device)
    #     self.vit.pos_embed = self.vit.pos_embed.to(device)
        
    #     x = self.vit.patch_embed(x)  # (B, num_patches, hidden_dim)
    #     cls_tokens = self.vit.cls_token.expand(B, -1, -1).to(device)  # Move to device
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = x + self.vit.pos_embed
    #     x = self.vit.pos_drop(x)

    #     for blk in self.vit.blocks:
    #         blk = blk.to(device)  # Ensure block is on the right device
    #         x = blk(x)

    #     x = self.vit.norm(x)
    #     return x  # (B, num_patches+1, hidden_dim)
    def vit_forward(self, x):
        # Use DINOv2's built-in forward_features (handles patch embedding, cls token, pos embedding, etc.)
        features_dict = self.vit.forward_features(x)  # Returns a dictionary
        
        # Extract patch embeddings (shape: B, num_patches, hidden_dim)
        patch_embeddings = features_dict["x_norm_patchtokens"]
        
        # Extract [CLS] token (shape: B, hidden_dim) if needed
        #cls_token = features_dict["x_norm_clstoken"]
        
        return patch_embeddings  # Return patch embeddings (for segmentation tasks)
    
    def forward(self, x):
        device = x.device  # Get the input tensor's device
        B, C, H, W = x.size()
        features_list = []

        # Process each channel through ViT
        for c in range(C):
            x_c = x[:, c].unsqueeze(1).to(device)  # Move input to the correct device
            x_c = self.replicate_channel_to_rgb(x_c)
            patch_embeddings = self.vit.forward_features(x_c.to(device))["x_norm_patchtokens"]
            B, num_patches, hidden_dim = patch_embeddings.shape
            h = w = int(num_patches ** 0.5)  # e.g., 16x16 for 224x224 image with patch_size=14
            features = patch_embeddings.permute(0, 2, 1).view(B, hidden_dim, h, w)
            
            features_list.append(features)

        x = torch.cat(features_list, dim=1)  # (B, C*768, 14, 14)
        
        # Apply attention mechanisms
        x, attn = self.eca(x)
        x = self.spatial_gate(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

        if self.cut_at_pooling:
            return x, attn

        # Feature normalization
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if not self.training:
            bn_x = F.normalize(bn_x)
            return bn_x, attn

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            return self.classifier(bn_x), attn
        return bn_x, attn
    

class ECALayer(nn.Module):
    def __init__(self, channel, k_size=None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not k_size:
            t = int(abs(math.log2(channel) / 2 + 0.5))
            k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(1, 2))
        y = y.transpose(1, 2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x), y

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size=7, padding=3, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([x.max(dim=1)[0].unsqueeze(1), x.mean(dim=1).unsqueeze(1)], dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=not bn)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x
    


def vitbase(**kwargs):
    return ViTBasedModel(**kwargs)