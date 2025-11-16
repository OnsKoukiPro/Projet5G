import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict  # ADD THIS IMPORT
from einops import rearrange

class IntermediateFusion(nn.Module):
    """Intermediate Fusion with Cross-Attention"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.feature_dim = config['feature_dim']
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']

        # Cross-attention layers
        self.infra_self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        self.vehicle_self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)
        self.norm3 = nn.LayerNorm(self.feature_dim)

    def forward(self, infra_features: torch.Tensor, vehicle_features: torch.Tensor) -> torch.Tensor:
        batch_size = infra_features.shape[0]

        # Reshape features for attention [B, C, H, W] -> [B, H*W, C]
        infra_features = rearrange(infra_features, 'b c h w -> b (h w) c')
        vehicle_features = rearrange(vehicle_features, 'b c h w -> b (h w) c')

        # Self-attention on individual features
        infra_self_attn, _ = self.infra_self_attention(
            infra_features, infra_features, infra_features
        )
        infra_features = self.norm1(infra_features + infra_self_attn)

        vehicle_self_attn, _ = self.vehicle_self_attention(
            vehicle_features, vehicle_features, vehicle_features
        )
        vehicle_features = self.norm2(vehicle_features + vehicle_self_attn)

        # Cross-attention between modalities
        cross_attn, attention_weights = self.cross_attention(
            query=infra_features,
            key=vehicle_features,
            value=vehicle_features
        )

        # Residual connection and normalization
        fused_features = self.norm3(infra_features + cross_attn)

        # Feature projection
        fused_features = self.feature_proj(fused_features)

        # Reshape back to original format [B, H*W, C] -> [B, C, H, W]
        original_shape = infra_features.shape
        fused_features = rearrange(fused_features, 'b (h w) c -> b c h w',
                                 h=int(original_shape[1]**0.5))

        return fused_features