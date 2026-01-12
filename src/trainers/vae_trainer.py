"""
VAE Trainer

Trainer for Variational Autoencoder with discriminator and perceptual loss.
"""

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torchvision.utils import make_grid
import os
from typing import Dict, Any

from .base import BaseTrainer
from models.discriminator import Discriminator
from models.lpips import LPIPS


class VAETrainer(BaseTrainer):
    """Trainer for VAE with adversarial and perceptual losses.

    Implements training loop with:
    - Reconstruction loss (MSE)
    - KL divergence loss
    - Adversarial loss (with discriminator)
    - Perceptual loss (LPIPS)
    """

    def __init__(self, model, config: Dict[str, Any], device: str = "cuda"):
        """Initialize VAE trainer.

        Args:
            model: VAE model
            config: Configuration dictionary
            device: Device to train on
        """
        super().__init__(model, config, device)

        # Get dataset config
        from src.datasets.registry import get_dataset_config
        dataset_name = config.get("dataset")
        dataset_cfg = get_dataset_config(dataset_name)

        # Create discriminator
        self.discriminator = Discriminator(
            im_channels=dataset_cfg.get("im_channels", 3)
        ).to(self.device)

        # Create LPIPS perceptual loss model (frozen)
        self.lpips_model = LPIPS().eval().to(self.device)

        # Discriminator optimizer
        self.optimizer_d = Adam(
            self.discriminator.parameters(),
            lr=self.train_cfg.get("lr", 1e-5),
            betas=(0.5, 0.999)
        )

        # Loss weights and configuration
        self.disc_start = self.train_cfg.get("disc_start", 7500)
        self.disc_weight = self.train_cfg.get("disc_weight", 0.5)
        self.perceptual_weight = self.train_cfg.get("perceptual_weight", 1.0)
        self.kl_weight = self.train_cfg.get("kl_weight", 0.000005)
        self.acc_steps = self.train_cfg.get("autoencoder_acc_steps", 1)
        self.image_save_steps = self.train_cfg.get("autoencoder_img_save_steps", 64)

        # Loss functions
        self.recon_criterion = nn.MSELoss()
        self.disc_criterion = nn.MSELoss()

        # Image saving
        self.img_save_count = 0
        os.makedirs(os.path.join(self.output_dir, "vae_autoencoder_samples"), exist_ok=True)

        # Load checkpoints if they exist
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self):
        """Load existing VAE and discriminator checkpoints if available."""
        vae_ckpt = os.path.join(
            self.output_dir,
            self.train_cfg.get("vae_autoencoder_ckpt_name", "vae_autoencoder.pth")
        )
        disc_ckpt = os.path.join(
            self.output_dir,
            self.train_cfg.get("vae_discriminator_ckpt_name", "vae_discriminator.pth")
        )

        if os.path.exists(vae_ckpt):
            self.model.load_state_dict(torch.load(vae_ckpt, map_location=self.device))
            print(f"Loaded autoencoder from {vae_ckpt}")

        if os.path.exists(disc_ckpt):
            self.discriminator.load_state_dict(torch.load(disc_ckpt, map_location=self.device))
            print(f"Loaded discriminator from {disc_ckpt}")

    def train_step(self, batch) -> Dict[str, float]:
        """Execute single VAE training step.

        Args:
            batch: Batch of images

        Returns:
            Dictionary of loss values
        """
        images = batch.to(self.device)

        # Forward pass through VAE
        output, encoder_output = self.model(images)

        # Save images periodically
        if self.global_step % self.image_save_steps == 0 or self.global_step == 1:
            self._save_images(images, output)

        ######### Optimize Generator (VAE) ##########
        # Reconstruction loss
        recon_loss = self.recon_criterion(output, images)

        # KL divergence loss
        mean, logvar = torch.chunk(encoder_output, 2, dim=1)
        kl_loss = torch.mean(
            0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1 - logvar, dim=[1, 2, 3])
        )

        # Total generator loss
        g_loss = recon_loss + self.kl_weight * kl_loss

        # Adversarial loss (after disc_start steps)
        disc_fake_loss_val = 0.0
        if self.global_step > self.disc_start:
            disc_fake_pred = self.discriminator(output)
            disc_fake_loss = self.disc_criterion(
                disc_fake_pred,
                torch.ones(disc_fake_pred.shape, device=self.device)
            )
            disc_fake_loss_val = disc_fake_loss.item()
            g_loss += self.disc_weight * disc_fake_loss

        # Perceptual loss (LPIPS)
        lpips_loss = torch.mean(self.lpips_model(output, images))
        g_loss += self.perceptual_weight * lpips_loss

        # Backward and optimize generator
        self.optimizer.zero_grad()
        g_loss.backward()
        self.optimizer.step()

        ######### Optimize Discriminator #######
        d_loss_val = 0.0
        if self.global_step > self.disc_start:
            disc_real_pred = self.discriminator(images)
            disc_fake_pred = self.discriminator(output.detach())

            disc_real_loss = self.disc_criterion(
                disc_real_pred,
                torch.ones(disc_real_pred.shape, device=self.device)
            )
            disc_fake_loss = self.disc_criterion(
                disc_fake_pred,
                torch.zeros(disc_fake_pred.shape, device=self.device)
            )

            d_loss = self.disc_weight * (disc_real_loss + disc_fake_loss) / 2
            d_loss_val = d_loss.item()

            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

        return {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "lpips_loss": lpips_loss.item(),
            "g_adv_loss": disc_fake_loss_val,
            "d_loss": d_loss_val,
            "total_loss": g_loss.item()
        }

    def _save_images(self, images, output):
        """Save sample images showing input vs reconstruction.

        Args:
            images: Input images
            output: Reconstructed images
        """
        sample_size = min(8, images.shape[0])
        save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
        save_output = (save_output + 1) / 2
        save_input = ((images[:sample_size] + 1) / 2).detach().cpu()

        grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
        img = torchvision.transforms.ToPILImage()(grid)

        save_path = os.path.join(
            self.output_dir,
            'vae_autoencoder_samples',
            f'current_autoencoder_sample_{self.img_save_count}.png'
        )
        img.save(save_path)
        img.close()
        self.img_save_count += 1

    def save_checkpoint(self, filename=None):
        """Save VAE and discriminator checkpoints.

        Args:
            filename: Optional custom filename
        """
        # Save VAE
        vae_path = os.path.join(
            self.output_dir,
            self.train_cfg.get("vae_autoencoder_ckpt_name", "vae_autoencoder.pth")
        )
        torch.save(self.model.state_dict(), vae_path)

        # Save discriminator
        disc_path = os.path.join(
            self.output_dir,
            self.train_cfg.get("vae_discriminator_ckpt_name", "vae_discriminator.pth")
        )
        torch.save(self.discriminator.state_dict(), disc_path)

        print(f"Saved VAE checkpoint: {vae_path}")
        print(f"Saved discriminator checkpoint: {disc_path}")

    def validation_step(self, batch) -> Dict[str, float]:
        """Execute VAE validation step.

        Args:
            batch: Batch of images

        Returns:
            Dictionary of validation metrics
        """
        images = batch.to(self.device)
        output, encoder_output = self.model(images)

        recon_loss = self.recon_criterion(output, images)

        return {"val_recon_loss": recon_loss.item()}
