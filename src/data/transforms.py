import numpy as np
import torch
from typing import Dict, List, Optional
import numba

class Compose:
    """Compose multiple transforms"""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, data_dict: Dict) -> Dict:
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict

class RandomRotation:
    """Random rotation augmentation"""

    def __init__(self, rotation_range: List[float] = [-np.pi/4, np.pi/4]):
        self.rotation_range = rotation_range

    def __call__(self, data_dict: Dict) -> Dict:
        if np.random.random() > 0.5:  # 50% chance to apply
            angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])

            # Rotate infrastructure points
            if 'infra_points' in data_dict:
                infra_points = data_dict['infra_points'].numpy()
                infra_points[:, :2] = self._rotate_points(infra_points[:, :2], angle)
                data_dict['infra_points'] = torch.from_numpy(infra_points).float()

            # Rotate vehicle points
            if 'vehicle_points' in data_dict:
                vehicle_points = data_dict['vehicle_points'].numpy()
                vehicle_points[:, :2] = self._rotate_points(vehicle_points[:, :2], angle)
                data_dict['vehicle_points'] = torch.from_numpy(vehicle_points).float()

            # Rotate ground truth boxes
            for key in ['infra_gt_boxes', 'vehicle_gt_boxes']:
                if key in data_dict and len(data_dict[key]) > 0:
                    boxes = data_dict[key].numpy()
                    boxes[:, 6] += angle  # Update rotation
                    data_dict[key] = torch.from_numpy(boxes).float()

        return data_dict

    def _rotate_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        """Rotate points by given angle"""
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        return points @ rotation_matrix.T

class RandomFlip:
    """Random flip augmentation"""

    def __init__(self, flip_prob: float = 0.5):
        self.flip_prob = flip_prob

    def __call__(self, data_dict: Dict) -> Dict:
        # Flip along x-axis
        if np.random.random() < self.flip_prob:
            if 'infra_points' in data_dict:
                infra_points = data_dict['infra_points'].numpy()
                infra_points[:, 1] = -infra_points[:, 1]  # Flip y-coordinate
                data_dict['infra_points'] = torch.from_numpy(infra_points).float()

            if 'vehicle_points' in data_dict:
                vehicle_points = data_dict['vehicle_points'].numpy()
                vehicle_points[:, 1] = -vehicle_points[:, 1]  # Flip y-coordinate
                data_dict['vehicle_points'] = torch.from_numpy(vehicle_points).float()

            # Flip ground truth boxes
            for key in ['infra_gt_boxes', 'vehicle_gt_boxes']:
                if key in data_dict and len(data_dict[key]) > 0:
                    boxes = data_dict[key].numpy()
                    boxes[:, 1] = -boxes[:, 1]  # Flip y-coordinate
                    boxes[:, 6] = -boxes[:, 6]  # Flip rotation
                    data_dict[key] = torch.from_numpy(boxes).float()

        return data_dict

class RandomScaling:
    """Random scaling augmentation"""

    def __init__(self, scale_range: List[float] = [0.95, 1.05]):
        self.scale_range = scale_range

    def __call__(self, data_dict: Dict) -> Dict:
        if np.random.random() > 0.5:  # 50% chance to apply
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])

            # Scale points
            for key in ['infra_points', 'vehicle_points']:
                if key in data_dict:
                    points = data_dict[key].numpy()
                    points[:, :3] *= scale_factor
                    data_dict[key] = torch.from_numpy(points).float()

            # Scale ground truth boxes
            for key in ['infra_gt_boxes', 'vehicle_gt_boxes']:
                if key in data_dict and len(data_dict[key]) > 0:
                    boxes = data_dict[key].numpy()
                    boxes[:, :3] *= scale_factor  # Position
                    boxes[:, 3:6] *= scale_factor  # Dimensions
                    data_dict[key] = torch.from_numpy(boxes).float()

        return data_dict

# For backward compatibility
def build_transforms(config: Optional[Dict] = None):
    """Build transform pipeline from config"""
    if config is None:
        return None

    transforms = []

    if config.get('random_rotation', True):
        transforms.append(RandomRotation())

    if config.get('random_flip', True):
        transforms.append(RandomFlip())

    if config.get('random_scaling', True):
        transforms.append(RandomScaling())

    return Compose(transforms) if transforms else None