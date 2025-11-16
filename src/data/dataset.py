import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import open3d as o3d

from .preprocessing import DataPreprocessor
from .transforms import Compose, RandomRotation, RandomFlip, RandomScaling

class DAIRV2XDataset(Dataset):
    """DAIR-V2X Cooperative Perception Dataset"""

    def __init__(self,
                 root_path: str,
                 split: str = 'train',
                 config: Dict = None,
                 transform: Optional[Compose] = None):

        self.root_path = Path(root_path)
        self.split = split
        self.config = config
        self.transform = transform

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(config)

        # Load data samples
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} {split} samples")

    def _load_samples(self) -> List[Dict]:
        """Load and validate data samples"""
        samples = []

        # Infrastructure data
        infra_lidar_dir = self.root_path / "infrastructure-side" / "velodyne"
        infra_label_dir = self.root_path / "infrastructure-side" / "label" / "virtuallidar"

        # Vehicle data
        vehicle_lidar_dir = self.root_path / "vehicle-side" / "velodyne"
        vehicle_label_dir = self.root_path / "vehicle-side" / "label" / "camera"

        # Get synchronized pairs
        infra_files = sorted(infra_lidar_dir.glob("*.bin"))

        for infra_lidar_path in infra_files:
            sample_id = infra_lidar_path.stem

            # Find corresponding vehicle data
            vehicle_lidar_path = vehicle_lidar_dir / f"{sample_id}.bin"
            infra_label_path = infra_label_dir / f"{sample_id}.json"
            vehicle_label_path = vehicle_label_dir / f"{sample_id}.json"

            # Check if all files exist
            if (vehicle_lidar_path.exists() and
                infra_label_path.exists() and
                vehicle_label_path.exists()):

                samples.append({
                    'sample_id': sample_id,
                    'infra_lidar_path': str(infra_lidar_path),
                    'vehicle_lidar_path': str(vehicle_lidar_path),
                    'infra_label_path': str(infra_label_path),
                    'vehicle_label_path': str(vehicle_label_path)
                })

        # Split data
        split_idx = self._get_split_indices(len(samples))
        samples = [samples[i] for i in split_idx]

        return samples

    def _get_split_indices(self, total_samples: int) -> List[int]:
        """Get indices for train/val/test split"""
        train_ratio = self.config['dataset']['train_split']
        val_ratio = self.config['dataset']['val_split']

        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        if self.split == 'train':
            return list(range(0, train_end))
        elif self.split == 'val':
            return list(range(train_end, val_end))
        else:  # test
            return list(range(val_end, total_samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]

        try:
            # Load point clouds
            infra_points = self._load_lidar(sample_info['infra_lidar_path'])
            vehicle_points = self._load_lidar(sample_info['vehicle_lidar_path'])

            # Load labels
            infra_labels = self._load_labels(sample_info['infra_label_path'])
            vehicle_labels = self._load_labels(sample_info['vehicle_label_path'])

            # Preprocess data
            data_dict = self.preprocessor(
                infra_points=infra_points,
                vehicle_points=vehicle_points,
                infra_labels=infra_labels,
                vehicle_labels=vehicle_labels
            )

            # Apply transforms for training
            if self.transform and self.split == 'train':
                data_dict = self.transform(data_dict)

            data_dict['sample_id'] = sample_info['sample_id']
            return data_dict

        except Exception as e:
            print(f"Error loading sample {sample_info['sample_id']}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

    def _load_lidar(self, path: str) -> np.ndarray:
        """Load LiDAR point cloud from .bin file"""
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return points

    def _load_labels(self, path: str) -> List[Dict]:
        """Load labels from JSON file"""
        with open(path, 'r') as f:
            labels = json.load(f)

        # Handle different label formats
        if isinstance(labels, list):
            return labels
        elif isinstance(labels, dict) and 'objects' in labels:
            return labels['objects']
        else:
            return []