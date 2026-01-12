"""
Trainers for nanoVLA-RL models

Provides trainer classes for VAE, DiT, and other models.
"""

from .base import BaseTrainer
from .vae_trainer import VAETrainer

__all__ = [
    'BaseTrainer',
    'VAETrainer'
]
