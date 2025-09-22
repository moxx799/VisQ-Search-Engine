from .unet2 import unet
import torch
import torch.nn as nn
import torch.nn.functional as F
from clustercontrast.utils.serialization import load_checkpoint


class MoECrossAttention(nn.Module):
    def __init__(self, output_dim=4608, dropout=0.1, num_features=0,new_in_channels=0,**args):
        super(MoECrossAttention, self).__init__()

        self.gate_input_dim = 3 * output_dim
        # Gating network with dropout for stability
        self.gating = nn.Sequential(
            nn.Linear(self.gate_input_dim, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        # Learnable query vector for cross-attention
        self.query = nn.Parameter(torch.randn(output_dim))
        
        # Dropout layer for attention scores
        self.attn_dropout = nn.Dropout(dropout)
        
    def _load_expert(self, path, num_channels):
        expert = unet(img_size=50, num_features=0, new_in_channels=num_channels, norm=True, dropout=0,
                          num_classes=0, pooling_type='gem')
        checkpoint = load_checkpoint(path)
        expert.load_state_dict(checkpoint['state_dict'])
        return expert
    
    def forward(self, e1,e2,e3):
        # Get outputs from each expert
 
        # Concatenate outputs for gating input
        gate_input = torch.cat([e1, e2, e3], dim=-1)
        weights = self.gating(gate_input)  # (batch_size, 3)
        
        # Scale each expert's output by its weight
        w1, w2, w3 = weights[:, 0], weights[:, 1], weights[:, 2]
        e1 = e1 * w1.unsqueeze(-1)
        e2 = e2 * w2.unsqueeze(-1)
        e3 = e3 * w3.unsqueeze(-1)
        
        # Stack scaled outputs as keys/values (batch_size, 3, output_dim)
        keys_values = torch.stack([e1, e2, e3], dim=1)
        
        # Expand learned query to match batch size (batch_size, 1, output_dim)
        query = self.query.unsqueeze(0).unsqueeze(1).repeat(x.size(0), 1, 1)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, keys_values.transpose(1, 2)) / (self.query.size(-1) ** 0.5)
        attn_scores = self.attn_dropout(attn_scores)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, keys_values).squeeze(1)
        return output
        
        
        
def moe(**kwargs):
    return MoECrossAttention(**kwargs)