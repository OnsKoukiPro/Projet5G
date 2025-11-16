import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import wandb
import os
from typing import Dict, List

from ..models import CooperativePointPillars
from ..data import DAIRV2XDataset
from .loss import DetectionLoss

class CooperativeTrainer:
    """Training class for cooperative perception"""

    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device

        # Initialize model
        self.model = CooperativePointPillars(config).to(device)

        # Loss function
        self.criterion = DetectionLoss(config)

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs']
        )

        # Metrics
        self.best_map = 0.0
        self.train_losses = []
        self.val_metrics = []

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            batch_data = self._move_to_device(batch_data)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_data)

            # Compute loss
            loss_dict = self.criterion(predictions, batch_data)
            total_loss_value = sum(loss_dict.values())

            # Backward pass
            total_loss_value.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )

            self.optimizer.step()

            # Update progress bar
            total_loss += total_loss_value.item()
            pbar.set_postfix({
                'loss': f'{total_loss_value.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'train/loss': total_loss_value.item(),
                    'train/classification_loss': loss_dict['cls_loss'].item(),
                    'train/localization_loss': loss_dict['loc_loss'].item(),
                    'train/epoch': epoch
                })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc='Validation'):
                batch_data = self._move_to_device(batch_data)
                predictions = self.model(batch_data)

                # Collect predictions and targets for evaluation
                all_predictions.append(self._process_predictions(predictions))
                all_targets.append(self._process_targets(batch_data))

        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)

        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'val/mAP': metrics['mAP'],
                'val/AP_0.5': metrics['AP_0.5'],
                'val/AP_0.7': metrics['AP_0.7'],
                'val/epoch': epoch
            })

        # Save best model
        if metrics['mAP'] > self.best_map:
            self.best_map = metrics['mAP']
            self._save_checkpoint(epoch, metrics, is_best=True)

        self.val_metrics.append(metrics)
        return metrics

    def _move_to_device(self, batch_data: Dict) -> Dict:
        """Move batch data to device"""
        device_data = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                device_data[key] = value.to(self.device)
            else:
                device_data[key] = value
        return device_data

    def _process_predictions(self, predictions: Dict) -> Dict:
        """Process model predictions for evaluation"""
        # Implementation for converting raw predictions to detection format
        pass

    def _process_targets(self, batch_data: Dict) -> Dict:
        """Process ground truth for evaluation"""
        # Implementation for preparing targets for metrics computation
        pass

    def _compute_metrics(self, predictions: List, targets: List) -> Dict:
        """Compute evaluation metrics"""
        # Implementation of mAP, AP@0.5, AP@0.7 calculations
        return {
            'mAP': 0.0,
            'AP_0.5': 0.0,
            'AP_0.7': 0.0,
            'Recall': 0.0,
            'Precision': 0.0
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'metrics': metrics,
            'config': self.config
        }

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"

        torch.save(checkpoint, f"/workspace/checkpoints/{filename}")

        print(f"Checkpoint saved: {filename} with mAP: {metrics['mAP']:.4f}")