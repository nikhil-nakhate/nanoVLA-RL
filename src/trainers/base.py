"""
Base Trainer Class

Abstract base class for all model trainers providing common training loop functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
import os


class BaseTrainer(ABC):
    """Abstract base trainer for all model types.

    Provides common training loop, checkpointing, and logging functionality.
    Subclasses must implement train_step and validation_step methods.
    """

    def __init__(self, model, config: Dict[str, Any], device: str = "cuda"):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            device: Device to train on ('cuda', 'cpu', or 'mps')
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Extract train config
        self.train_cfg = config.get("train", {})
        self.num_epochs = self.train_cfg.get("epochs", 100)
        self.batch_size = self.train_cfg.get("batch_size", 8)
        self.lr = self.train_cfg.get("lr", 1e-4)
        self.save_every = self.train_cfg.get("save_every", 100)
        self.output_dir = self.train_cfg.get("output_dir", "outputs")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize optimizer (subclasses can override)
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        """Build optimizer from config.

        Can be overridden by subclasses for custom optimizers.

        Returns:
            PyTorch optimizer
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.train_cfg.get("weight_decay", 0.01)
        )

    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]:
        """Execute single training step.

        Args:
            batch: Batch of data from dataloader

        Returns:
            Dictionary of loss values
        """
        pass

    def validation_step(self, batch) -> Dict[str, float]:
        """Execute single validation step.

        Optional method - can be overridden by subclasses.
        Default implementation returns empty dict.

        Args:
            batch: Batch of data from dataloader

        Returns:
            Dictionary of validation metrics
        """
        return {}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_losses = []

            for batch in train_loader:
                losses = self.train_step(batch)
                self.global_step += 1

                epoch_losses.append(losses)

                # Periodic logging
                if self.global_step % 10 == 0:
                    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Step {self.global_step}: {loss_str}")

                # Periodic checkpointing
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint()

            # Epoch summary
            avg_losses = {}
            if epoch_losses:
                keys = epoch_losses[0].keys()
                for key in keys:
                    avg_losses[key] = sum(d[key] for d in epoch_losses) / len(epoch_losses)

            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            print(f"Epoch {epoch+1} complete. Avg losses: {loss_str}")

            # Validation
            if val_loader:
                self.validate(val_loader)

            # Save checkpoint at end of epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        print("Training complete!")

    def validate(self, val_loader: DataLoader):
        """Run validation.

        Args:
            val_loader: Validation data loader
        """
        self.model.eval()
        val_metrics = []

        with torch.no_grad():
            for batch in val_loader:
                metrics = self.validation_step(batch)
                if metrics:
                    val_metrics.append(metrics)

        if val_metrics:
            # Aggregate metrics
            keys = val_metrics[0].keys()
            avg_metrics = {}
            for key in keys:
                avg_metrics[key] = sum(d[key] for d in val_metrics) / len(val_metrics)

            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            print(f"Validation: {metric_str}")

    def save_checkpoint(self, filename: Optional[str] = None):
        """Save model checkpoint.

        Args:
            filename: Optional custom filename. If None, uses default naming.
        """
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"

        checkpoint_path = os.path.join(self.output_dir, filename)

        checkpoint = {
            "step": self.global_step,
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        # Save to temp file first, then rename (atomic operation)
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")
