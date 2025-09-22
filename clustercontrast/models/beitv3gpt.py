import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BeitModel, BeitConfig
from torch.nn import init



class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2).contiguous()).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x), y

def prep_img(xc):
    [B, W, H] = xc.size()
    xc = xc.unsqueeze(1) 
    xc = xc.repeat(1, 3, 1, 1)
    return xc  


class BEiTv3Encoder(nn.Module):
    def __init__(self, depth=0, pretrained=True, cut_at_pooling=False, img_size=224,
                 num_features=0, new_in_channels=7, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(BEiTv3Encoder, self).__init__()
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.new_in_channels = new_in_channels
        self.img_size = img_size
        self.num_features = num_features

        config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        config.image_size = img_size
        # (Optional) we only need last_hidden_state
        config.output_hidden_states = False
        self.beit = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k", config=config)

        # Freeze first 9 layers
        for i, layer in enumerate(self.beit.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = i >= 9
        for p in self.beit.embeddings.parameters():
            p.requires_grad = False

        self.feature_dim = 768       # d
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2  # n
        self.output_size = 3

        # NEW: ECA over the token-grid channels (d)
        self.token_eca = eca_layer(channel=self.feature_dim, k_size=3)

        # Keep your downstream pieces the same
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        self.channel_reduction = nn.Conv2d(self.feature_dim, 512, kernel_size=1)

        # This ECA was originally over concatenated channels; keep if you still want it
        self.eca = eca_layer(new_in_channels)

        self.final_feat_dim = 512 * self.output_size * self.output_size

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

    def _tokens_to_grid(self, token_feats: torch.Tensor) -> torch.Tensor:
        """
        token_feats: [B, n, d] (CLS already removed)
        returns:     [B, d, S, S] where S = sqrt(n)
        """
        B, n, d = token_feats.shape
        S = int(math.isqrt(n))
        if S * S != n:
            # This should not happen with square images & square patching, but guard anyway
            # Pad to next square then crop back after reshape
            nextS = S + 1
            pad_n = nextS * nextS - n
            pad = torch.zeros(B, pad_n, d, dtype=token_feats.dtype, device=token_feats.device)
            token_feats = torch.cat([token_feats, pad], dim=1)
            grid = token_feats.view(B, nextS, nextS, d).permute(0, 3, 1, 2)[:, :, :S, :S]
        else:
            grid = token_feats.view(B, S, S, d).permute(0, 3, 1, 2)  # [B, d, S, S]
        return grid

    def forward(self, x):
        B, C, H, W = x.shape
        processed_features = []
        attn_maps = []

        for i in range(C):
            x_channel = x[:, i:i+1, :, :]
            x_channel_3c = x_channel.repeat(1, 3, 1, 1)

            outputs = self.beit(x_channel_3c)                        # last_hidden_state: [B, 1+n, d]
            seq = outputs.last_hidden_state                          # include CLS at index 0
            tokens = seq[:, 1:, :]                                   # drop CLS => [B, n, d]
            grid = self._tokens_to_grid(tokens)                      # [B, d, S, S]

            # >>> this is the bit you asked for: use the whole encoding (n×d) reshaped to √n×√n×d and send to ECA
            grid_eca, token_eca_weights = self.token_eca(grid)       # both [B, d, S, S]
            attn_maps.append(token_eca_weights)                      # keep if you want to inspect

            # Continue with your original pipeline (downsample + reduce channels)
            features = self.adaptive_pool(grid_eca)                  # [B, d, 3, 3]
            features = self.channel_reduction(features)              # [B, 512, 3, 3]
            processed_features.append(features)

        x = torch.cat(processed_features, dim=1)                     # [B, C*512, 3, 3]
        x, attn_weights = self.eca(x)                                # channel attention across channels from all inputs

        x = x.reshape(B, C, -1)                                      # [B, C, 512*3*3]
        x = torch.sum(x, dim=1)                                      # [B, 512*3*3]

        bn_x = self.feat_bn(x)

        if not self.training:
            bn_x = F.normalize(bn_x)
            # Optionally return both ECA maps (token-level and late-stage)
            return bn_x, {"token_eca": attn_maps, "late_eca": attn_weights}

        if self.norm:
            bn_x = F.normalize(bn_x)
        if self.dropout > 0:
            bn_x = self.drop(bn_x)
        if self.num_classes > 0:
            return self.classifier(bn_x), {"token_eca": attn_maps, "late_eca": attn_weights}
        return bn_x, {"token_eca": attn_maps, "late_eca": attn_weights}
    
    
    
def beit_encoder(**kwargs):
    model = BEiTv3Encoder(**kwargs)
    return model
