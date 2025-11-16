import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict  # ADD THIS IMPORT

class MultiGroupHead(nn.Module):
    """Multi-group detection head"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Simplified detection head for testing
        self.cls_head = nn.Conv2d(256, 3, 1)  # 3 classes
        self.reg_head = nn.Conv2d(256, 7, 1)  # 7 box parameters (x, y, z, l, w, h, θ)

    def forward(self, features: torch.Tensor) -> Dict:
        """Forward pass through detection head"""
        cls_preds = self.cls_head(features)
        box_preds = self.reg_head(features)

        return {
            'cls_preds': cls_preds,
            'box_preds': box_preds
        }