#!/usr/bin/env python3

import sys
import os

# CRITICAL: Add workspace to Python path FIRST
sys.path.insert(0, '/workspace')

# Now import other modules
import yaml
import torch
from torch.utils.data import DataLoader

try:
    from src.data import DAIRV2XDataset
    print("✅ DAIRV2XDataset imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DAIRV2XDataset: {e}")
    sys.exit(1)

try:
    from src.training import CooperativeTrainer
    print("✅ CooperativeTrainer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CooperativeTrainer: {e}")
    sys.exit(1)

try:
    from src.utils.config import load_config
    print("✅ load_config imported successfully")
except ImportError as e:
    print(f"❌ Failed to import load_config: {e}")
    # Create a simple fallback
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    print("✅ Using fallback load_config")

def main():
    print("🚀 Starting Cooperative Perception Training")
    print("=" * 50)

    # Load configuration
    config_path = '/workspace/configs/base.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        # Create minimal config
        config = {
            'dataset': {
                'root_path': '/workspace/data/dair_v2x',
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'classes': ['Car', 'Pedestrian', 'Cyclist'],
                'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1],
                'voxel_size': [0.16, 0.16, 4],
                'max_points_per_voxel': 32,
                'max_voxels': 16000
            },
            'model': {
                'name': 'CooperativePointPillars',
                'backbone': {'type': 'PointPillars'},
                'fusion': {
                    'type': 'IntermediateFusion',
                    'feature_dim': 256,
                    'num_heads': 8,
                    'hidden_dim': 512
                },
                'head': {'type': 'MultiGroupHead'}
            },
            'training': {
                'batch_size': 2,
                'num_epochs': 5,
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'checkpoint_interval': 2,
                'gradient_clip': 10.0
            }
        }
        print("✅ Using minimal config")
    else:
        config = load_config(config_path)
        print("✅ Config loaded from file")

    # Initialize wandb (optional)
    try:
        import wandb
        wandb.init(
            project="cooperative-perception",
            config=config,
            name="intermediate_fusion_experiment"
        )
        use_wandb = True
        print("✅ Weights & Biases initialized")
    except ImportError:
        print("⚠️  wandb not available, continuing without logging")
        use_wandb = False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # Check data directory
    data_path = config['dataset']['root_path']
    if not os.path.exists(data_path):
        print(f"⚠️  Data path does not exist: {data_path}")
        print("📝 Creating dummy data structure for testing...")
        os.makedirs(data_path, exist_ok=True)
        # Create minimal directory structure
        for side in ['infrastructure-side', 'vehicle-side']:
            os.makedirs(f"{data_path}/{side}/velodyne", exist_ok=True)
            if side == 'infrastructure-side':
                os.makedirs(f"{data_path}/{side}/label/virtuallidar", exist_ok=True)
            else:
                os.makedirs(f"{data_path}/{side}/label/camera", exist_ok=True)

        # Create a dummy LiDAR file for testing
        import numpy as np
        dummy_points = np.random.randn(100, 4).astype(np.float32)
        dummy_points.tofile(f"{data_path}/infrastructure-side/velodyne/000001.bin")
        dummy_points.tofile(f"{data_path}/vehicle-side/velodyne/000001.bin")

        # Create dummy label files
        dummy_label = [{"type": "Car", "3d": {"center": {"x": 10, "y": 5, "z": 0}, "size": {"x": 4, "y": 2, "z": 1.5}, "rotation": {"z": 0.1}}}]
        import json
        with open(f"{data_path}/infrastructure-side/label/virtuallidar/000001.json", 'w') as f:
            json.dump(dummy_label, f)
        with open(f"{data_path}/vehicle-side/label/camera/000001.json", 'w') as f:
            json.dump(dummy_label, f)

        print("✅ Created dummy data for testing")

    try:
        # Create datasets
        print("📦 Creating datasets...")
        train_dataset = DAIRV2XDataset(
            root_path=config['dataset']['root_path'],
            split='train',
            config=config
        )

        val_dataset = DAIRV2XDataset(
            root_path=config['dataset']['root_path'],
            split='val',
            config=config
        )

        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Val dataset: {len(val_dataset)} samples")

        if len(train_dataset) == 0:
            print("❌ No training samples found!")
            return

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Start with 0 for stability
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Start with 0 for stability
            pin_memory=True
        )

        print("✅ Data loaders created")

        # Initialize trainer
        print("🎯 Initializing trainer...")
        trainer = CooperativeTrainer(config, device)
        print("✅ Trainer initialized")

        # Training loop
        num_epochs = config['training']['num_epochs']
        print(f"🔥 Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)
            print(f"📊 Training Loss: {train_loss:.4f}")

            # Validate
            if epoch % config['training']['checkpoint_interval'] == 0:
                metrics = trainer.validate(val_loader, epoch)
                print(f"📈 Validation mAP: {metrics['mAP']:.4f}")

            # Update learning rate
            trainer.scheduler.step()

        print("🎉 Training completed!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()