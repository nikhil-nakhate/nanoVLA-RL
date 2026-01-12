"""
Variational Autoencoder (VAE) model

Encoder-decoder architecture with latent space sampling for image generation.
"""

import torch
import torch.nn as nn
from models.blocks import DownBlock, MidBlock, UpBlock
from typing import Dict, Any


class VAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']

        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2 * self.z_channels, kernel_size=3, padding=1)

        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(2 * self.z_channels, 2 * self.z_channels, kernel_size=1)
        ####################################################

        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "VAE":
        """Build VAE from configuration dictionary.

        Config structure:
            dataset: <dataset_name>  # References dataset registry
            model:
                name: vae
                params:
                    z_channels: 4
                    down_channels: [128, 256, 384]
                    mid_channels: [384]
                    down_sample: [true, true]
                    attn_down: [false, false]
                    norm_channels: 32
                    num_heads: 4
                    num_down_layers: 2
                    num_mid_layers: 2
                    num_up_layers: 2
                weights: <optional_checkpoint_path>

        Args:
            cfg: Configuration dictionary

        Returns:
            Instantiated VAE model
        """
        from src.datasets.registry import get_dataset_config

        # Get dataset config
        dataset_name = cfg.get("dataset")
        if not dataset_name:
            raise ValueError("Config must include 'dataset' field")

        dataset_cfg = get_dataset_config(dataset_name)

        # Get model params
        model_cfg = cfg.get("model", {})
        params = model_cfg.get("params", {})

        # Build model
        model = cls(
            im_channels=dataset_cfg.get("im_channels", 3),
            model_config=params
        )

        # Load pretrained weights if specified
        weights_path = model_cfg.get("weights")
        if weights_path:
            import os
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded VAE weights from {weights_path}")
            else:
                print(f"Warning: Weights path {weights_path} does not exist")

        return model
