import numpy as np
import torch
from typing import Dict, List, Tuple
import numba

class DataPreprocessor:
    """Preprocess cooperative perception data"""

    def __init__(self, config: Dict):
        self.config = config
        self.point_cloud_range = np.array(config['dataset']['point_cloud_range'])
        self.voxel_size = np.array(config['dataset']['voxel_size'])
        self.max_points_per_voxel = config['dataset']['max_points_per_voxel']
        self.max_voxels = config['dataset']['max_voxels']
        self.class_names = config['dataset']['classes']

    def __call__(self,
                 infra_points: np.ndarray,
                 vehicle_points: np.ndarray,
                 infra_labels: List[Dict],
                 vehicle_labels: List[Dict]) -> Dict:

        # Filter points in range
        infra_points = self._filter_points(infra_points)
        vehicle_points = self._filter_points(vehicle_points)

        # Voxelize point clouds
        infra_voxels, infra_coords, infra_num_points = self._voxelize(infra_points)
        vehicle_voxels, vehicle_coords, vehicle_num_points = self._voxelize(vehicle_points)

        # Process labels
        infra_gt_boxes, infra_gt_labels = self._process_labels(infra_labels)
        vehicle_gt_boxes, vehicle_gt_labels = self._process_labels(vehicle_labels)

        # Convert to tensors
        data_dict = {
            # Infrastructure data
            'infra_points': torch.from_numpy(infra_points).float(),
            'infra_voxels': torch.from_numpy(infra_voxels).float(),
            'infra_coords': torch.from_numpy(infra_coords).int(),
            'infra_num_points': torch.from_numpy(infra_num_points).int(),

            # Vehicle data
            'vehicle_points': torch.from_numpy(vehicle_points).float(),
            'vehicle_voxels': torch.from_numpy(vehicle_voxels).float(),
            'vehicle_coords': torch.from_numpy(vehicle_coords).int(),
            'vehicle_num_points': torch.from_numpy(vehicle_num_points).int(),

            # Labels
            'infra_gt_boxes': torch.from_numpy(infra_gt_boxes).float(),
            'infra_gt_labels': torch.from_numpy(infra_gt_labels).long(),
            'vehicle_gt_boxes': torch.from_numpy(vehicle_gt_boxes).float(),
            'vehicle_gt_labels': torch.from_numpy(vehicle_gt_labels).long(),
        }

        return data_dict

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points within the defined range"""
        mask = ((points[:, 0] >= self.point_cloud_range[0]) &
                (points[:, 0] <= self.point_cloud_range[3]) &
                (points[:, 1] >= self.point_cloud_range[1]) &
                (points[:, 1] <= self.point_cloud_range[4]) &
                (points[:, 2] >= self.point_cloud_range[2]) &
                (points[:, 2] <= self.point_cloud_range[5]))
        return points[mask]

    def _voxelize(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxelize point cloud using optimized numba implementation"""
        return voxelize(points, self.voxel_size, self.point_cloud_range,
                       self.max_points_per_voxel, self.max_voxels)

    def _process_labels(self, labels: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Process labels into ground truth boxes and class labels"""
        gt_boxes = []
        gt_labels = []

        for label in labels:
            if '3d' not in label and '3d_box' not in label:
                continue

            box_data = label.get('3d', label.get('3d_box', {}))
            class_name = label.get('type', 'Car')

            if class_name not in self.class_names:
                continue

            # Extract box parameters
            center = np.array([
                box_data['center']['x'],
                box_data['center']['y'],
                box_data['center']['z']
            ])

            size = np.array([
                box_data['size']['x'],
                box_data['size']['y'],
                box_data['size']['z']
            ])

            rotation = box_data.get('rotation', {}).get('z', 0)

            # Create box representation [x, y, z, l, w, h, rotation]
            box = np.array([
                center[0], center[1], center[2],
                size[0], size[1], size[2], rotation
            ])

            gt_boxes.append(box)
            gt_labels.append(self.class_names.index(class_name))

        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
        else:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)

        return gt_boxes, gt_labels

@numba.jit(nopython=True)
def voxelize(points, voxel_size, point_cloud_range, max_points, max_voxels):
    """Optimized voxelization using numba"""
    # Implementation details for performance
    pass