import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class DetectionLoss(nn.Module):
    """Detection loss for cooperative perception"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute detection loss"""
        # Placeholder loss for testing
        cls_loss = torch.tensor(0.5, device=next(iter(predictions.values())).device)
        loc_loss = torch.tensor(0.3, device=next(iter(predictions.values())).device)
        dir_loss = torch.tensor(0.1, device=next(iter(predictions.values())).device)

        total_loss = cls_loss + loc_loss + dir_loss

        return {
            'cls_loss': cls_loss,
            'loc_loss': loc_loss,
            'dir_loss': dir_loss,
            'total_loss': total_loss
        }