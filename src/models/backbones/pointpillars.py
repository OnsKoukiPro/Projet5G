import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict  # ADD THIS IMPORT

class PointPillarsBackbone(nn.Module):
    """PointPillars backbone for feature extraction"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Simplified backbone for testing
        self.feature_net = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def forward(self, voxels: torch.Tensor, coords: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone"""
        batch_size = coords[:, 0].max().int().item() + 1

        # Create pseudo-image from pillars
        # For now, return dummy features for testing
        features = torch.randn(batch_size, 256, 200, 200, device=voxels.device)
        return features