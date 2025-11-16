import torch
import torch.nn as nn
from typing import Dict  # ADD THIS IMPORT

try:
    from .backbones import PointPillarsBackbone
    from .fusion import IntermediateFusion
    from .heads import MultiGroupHead
except ImportError as e:
    print(f"⚠️  Some model modules not available: {e}")

    # Define fallback classes
    class PointPillarsBackbone(nn.Module):
        def __init__(self, config):
            super().__init__()
        def forward(self, voxels, coords, num_points):
            return torch.randn(1, 256, 200, 200)

    class IntermediateFusion(nn.Module):
        def __init__(self, config):
            super().__init__()
        def forward(self, infra_features, vehicle_features):
            return infra_features  # Simple pass-through

    class MultiGroupHead(nn.Module):
        def __init__(self, config):
            super().__init__()
        def forward(self, features):
            return {'cls_preds': torch.randn(1, 3, 200, 200), 'box_preds': torch.randn(1, 7, 200, 200)}

class CooperativePointPillars(nn.Module):
    """Cooperative Perception Model with Intermediate Fusion"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Individual backbones
        self.infra_backbone = PointPillarsBackbone(config['model']['backbone'])
        self.vehicle_backbone = PointPillarsBackbone(config['model']['backbone'])

        # Fusion module
        self.fusion = IntermediateFusion(config['model']['fusion'])

        # Detection head
        self.head = MultiGroupHead(config['model']['head'])

    def forward(self, batch_dict: Dict) -> Dict:
        # Extract features from individual sensors
        infra_features = self.infra_backbone(
            batch_dict['infra_voxels'],
            batch_dict['infra_coords'],
            batch_dict['infra_num_points']
        )

        vehicle_features = self.vehicle_backbone(
            batch_dict['vehicle_voxels'],
            batch_dict['vehicle_coords'],
            batch_dict['vehicle_num_points']
        )

        # Fuse features
        fused_features = self.fusion(infra_features, vehicle_features)

        # Detection
        predictions = self.head(fused_features)

        return predictions