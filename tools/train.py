"""
Unified Training Script

Train nanoVLA-RL models using config-driven architecture.

Usage:
    python tools/train.py --config configs/vae.yaml
    python tools/train.py --config configs/dit.yaml --device cuda
"""

import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ops.config import load_config
from src.models import build_model_from_cfg
from src.trainers import VAETrainer
from src.datasets.registry import get_dataset_config
from datasets.celeb_dataset import CelebDataset


def get_trainer_class(model_name: str):
    """Get appropriate trainer class for model.

    Args:
        model_name: Name of the model (e.g., 'vae', 'dit')

    Returns:
        Trainer class

    Raises:
        ValueError: If no trainer exists for the model
    """
    trainers = {
        "vae": VAETrainer,
        # "dit": DITTrainer,  # To be added
    }

    if model_name not in trainers:
        raise ValueError(
            f"No trainer available for model '{model_name}'. "
            f"Available trainers: {list(trainers.keys())}"
        )

    return trainers[model_name]


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> str:
    """Get training device.

    Args:
        device_arg: Device argument from command line

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_arg == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif device_arg == "cpu":
        return "cpu"
    else:
        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Train nanoVLA-RL models with config-driven architecture"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., configs/vae.yaml)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to train on (default: auto-detect)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config, search_dir="configs")

    # Set random seed
    seed = config.get("train", {}).get("seed", 1111)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Build model
    print(f"\nBuilding model: {config['model']['name']}")
    model = build_model_from_cfg(config)
    print(f"Model built successfully")

    # Load dataset
    dataset_name = config.get("dataset")
    if not dataset_name:
        raise ValueError("Config must specify 'dataset' field")

    print(f"\nLoading dataset: {dataset_name}")
    dataset_cfg = get_dataset_config(dataset_name)

    # Create dataset
    train_dataset = CelebDataset(
        split="train",
        im_path=dataset_cfg["path"],
        im_size=dataset_cfg["im_size"],
        im_channels=dataset_cfg["im_channels"]
    )
    print(f"Dataset loaded: {len(train_dataset)} images")

    # Create dataloader
    train_cfg = config.get("train", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for performance
    )

    # Get trainer class
    model_name = config["model"]["name"]
    TrainerCls = get_trainer_class(model_name)

    # Initialize trainer
    print(f"\nInitializing {TrainerCls.__name__}")
    trainer = TrainerCls(model, config, device=device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "="*50)
    print("Starting training")
    print("="*50 + "\n")
    trainer.train(train_loader)

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()
