# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Critical**: Always use the `llm_agents` conda environment for all Python operations:
```bash
conda activate llm_agents
```

All Python commands, scripts, and terminal operations must run within this environment.

## Project Overview

This is a PyTorch-based project implementing multiple vision and generative models:

1. **VAE (Variational Autoencoder)** - Image compression and latent encoding
2. **DiT (Diffusion Transformer)** - Diffusion-based image generation in latent space
3. **PaliGemma** - Vision-language model combining SigLIP vision encoder with Gemma language model

The project follows a two-stage training approach for generative models:
- Stage 1: Train VAE to compress images into latent representations
- Stage 2: Train DiT in latent space for efficient diffusion-based generation

## Common Commands

### Training (New Config-Driven System)

**Unified Training Script** (Recommended):
```bash
# Train VAE
python tools/train.py --config configs/vae.yaml

# Train DiT (coming soon)
python tools/train.py --config configs/dit.yaml

# Specify device
python tools/train.py --config configs/vae.yaml --device cuda

# Resume from checkpoint
python tools/train.py --config configs/vae.yaml --resume outputs/checkpoint.pt
```

### Training (Legacy Scripts)

**Train VAE (Stage 1):**
```bash
python tools/train_vae.py --config configs/celebhq.yaml
```

**Train DiT in Latent Space (Stage 2):**
```bash
python tools/train_vae_dit.py --config configs/celebhq.yaml
```

**Note**: Legacy scripts still work but new code should use the unified training script.

### Inference

**VAE Reconstruction:**
```bash
python tools/infer_vae.py --config configs/celebhq.yaml
```

**DiT Image Sampling:**
```bash
python tools/sample_vae_dit.py --config configs/celebhq.yaml
```

**PaliGemma Vision-Language Inference:**
```bash
python tools/infer_paligemma.py \
  --model_path <path_to_hf_model> \
  --prompt "describe this image" \
  --image_file_path <path_to_image> \
  --max_tokens_to_generate 100
```

### Configuration

All model and training parameters are controlled via YAML configs in `configs/`. Key parameters:
- `dataset_params`: Image paths, size, channels
- `autoencoder_params`: VAE architecture (channels, layers, attention)
- `dit_params`: DiT architecture (patch size, layers, hidden size)
- `diffusion_params`: Diffusion process (timesteps, beta schedule)
- `train_params`: Training hyperparameters, checkpoint names, batch sizes

## Architecture

### VAE Architecture (`models/vae.py`)

Two-part encoder-decoder with separate paths:

**Encoder Pipeline:**
- `encoder_conv_in` → DownBlocks → MidBlocks → `encoder_norm_out` → `encoder_conv_out`
- Outputs mean and log-variance for latent distribution
- Reparameterization: `z = mean + std * noise`

**Decoder Pipeline:**
- `post_quant_conv` → `decoder_conv_in` → MidBlocks → UpBlocks → `decoder_norm_out` → `decoder_conv_out`
- Reconstructs image from latent sample

**Building Blocks** (`models/blocks.py`):
- `DownBlock`: Convolution layers with optional downsampling and attention
- `MidBlock`: Middle processing layers with attention
- `UpBlock`: Upsampling with transposed convolutions and optional attention

### DiT Architecture (`models/transformer.py`)

Diffusion model operating in latent space:

1. **Input**: Noisy latent `x` and timestep `t`
2. **Patch Embedding**: Convert latent into patches via `PatchEmbedding`
3. **Time Embedding**: Sinusoidal positional encoding for timestep
4. **Transformer Layers**: Stack of `TransformerLayer` with adaptive normalization (AdaLN)
5. **Output**: Predicted noise via unpatchify operation

**Key Feature**: Timestep conditioning is injected via adaptive layer normalization (scale and shift parameters) at each transformer layer.

### PaliGemma Architecture (`models/gemma.py`, `models/siglip.py`)

Multimodal architecture combining vision and language:

**Vision Tower** (SigLIP):
- Patch-based vision transformer
- Converts images to patch embeddings with position encodings
- Standard transformer encoder with self-attention and MLP layers

**Language Model** (Gemma):
- Decoder-only transformer with RoPE (Rotary Position Embeddings)
- Grouped Query Attention (GQA) for efficiency
- RMSNorm for layer normalization

**Integration**:
- `PaliGemmaMultiModalProjector`: Linear projection from vision to language embedding space
- `_merge_input_ids_with_image_features`: Merges image tokens with text tokens
- Special `image_token_index` (256000) marks where image embeddings are inserted

**KV Caching**: Implemented for efficient autoregressive generation during inference

### Training Losses

**VAE Training** (`tools/train_vae.py`):
- Reconstruction Loss (MSE)
- KL Divergence Loss (regularization)
- Perceptual Loss (LPIPS)
- Adversarial Loss (discriminator, starts after `disc_start` steps)

**DiT Training** (`tools/train_vae_dit.py`):
- Simple MSE between predicted noise and actual noise
- Samples random timesteps and noise for each batch

### Noise Scheduling (`scheduler/linear_scheduler.py`)

**LinearNoiseScheduler**: DDPM-style diffusion process
- `add_noise()`: Forward diffusion (add noise to clean image)
- `sample_prev_timestep()`: Reverse diffusion (denoise one step)
- Uses linear beta schedule from `beta_start` to `beta_end`

## Important Implementation Details

### Import Path Fixes (Phase 1 Complete)

**Fixed**: Import inconsistencies have been resolved. All code now uses correct plural paths:
- ✅ `from models.vae import VAE`
- ✅ `from models.transformer import DIT`
- ✅ `from models.patch_embed import PatchEmbedding`
- ✅ `from datasets.celeb_dataset import CelebDataset`

All legacy singular imports (`from model.*`, `from dataset.*`) have been updated to plural forms.

### Device Handling

All scripts include device detection with MPS support:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
```

### Dataset Requirements

The `CelebDataset` expects images in subdirectory structure:
- Path: `{im_path}/CelebA-HQ-img/*.{png,jpg,jpeg}`
- Images are center-cropped and resized to `im_size`
- Pixel values normalized to [-1, 1] range

### Checkpoint Management

Models save checkpoints to `{task_name}/` directory:
- VAE: `vae_autoencoder_ckpt.pth` and `vae_discriminator_ckpt.pth`
- DiT: `dit_ckpt.pth`
- Training resumes automatically if checkpoints exist

### Latent Caching

For faster DiT training, VAE can pre-encode images to latents:
- Set `save_latents: True` in config
- Latents saved to `{task_name}/{vae_latent_dir_name}/`
- DiT training checks for latents and loads them if available

## Config-Driven Architecture (Phase 2)

### Overview

The project now features a config-driven architecture inspired by open-slm-agents, enabling:
- Easy model swapping via configuration files
- Hierarchical config inheritance with deep merging
- Registry-based model and dataset management
- Unified training interface for all models
- Cleaner separation of concerns

### Directory Structure

```
nanoVLA-RL/
├── configs/                      # Configuration files
│   ├── datasets.yaml            # Dataset registry (central)
│   ├── base.yaml                # Base config (extended by others)
│   └── vae.yaml                 # VAE training config
│
├── src/                         # New organized source
│   ├── ops/
│   │   └── config.py           # Config loading with inheritance
│   ├── models/
│   │   ├── registry.py         # Model registration
│   │   ├── build.py            # Model builder
│   │   ├── vae.py              # VAE with from_config()
│   │   └── dit.py              # DiT with from_config()
│   ├── trainers/
│   │   ├── base.py             # BaseTrainer abstract class
│   │   └── vae_trainer.py      # VAETrainer
│   └── datasets/
│       └── registry.py         # Dataset registry loader
│
├── models/                      # Original models (still used)
├── datasets/                    # Original datasets (still used)
└── tools/
    ├── train.py                 # New unified training script
    ├── train_vae.py            # Legacy (still works)
    └── train_vae_dit.py        # Legacy (still works)
```

### Config System

**Hierarchical Configs with Inheritance:**

Configs support an `extends` keyword for inheritance. Child configs deep-merge with parents:

```yaml
# configs/base.yaml
train:
  seed: 1111
  batch_size: 4
  lr: 0.00001
  epochs: 100

# configs/vae.yaml
extends: base

dataset: celebhq  # References configs/datasets.yaml

model:
  name: vae
  params:
    z_channels: 4
    down_channels: [128, 256, 384]

train:
  epochs: 3  # Overrides base value
  lr: 0.0001  # Overrides base value
```

**Dataset Registry:**

Central `configs/datasets.yaml` maps names to paths:

```yaml
celebhq:
  path: 'data/CelebAMask-HQ'
  im_size: 128
  im_channels: 3
  type: image_folder
```

Configs reference datasets by name, eliminating path duplication.

### Model Registry

Models are registered using decorators and built from config:

```python
# Registration happens automatically via src/models/build.py
from src.models import build_model_from_cfg, list_models

# List available models
print(list_models())  # ['vae', 'dit']

# Build model from config
cfg = load_config('vae')
model = build_model_from_cfg(cfg)
```

Each model implements a `from_config()` classmethod:

```python
@classmethod
def from_config(cls, cfg: Dict[str, Any]) -> "VAE":
    """Build VAE from configuration dictionary."""
    from src.datasets.registry import get_dataset_config

    dataset_cfg = get_dataset_config(cfg['dataset'])
    model_cfg = cfg.get('model', {})

    return cls(
        im_channels=dataset_cfg['im_channels'],
        model_config=model_cfg['params']
    )
```

### Trainer System

**BaseTrainer Abstract Class:**

Provides common training loop functionality:
- Training/validation loops
- Checkpoint saving/loading (atomic writes)
- Logging and progress tracking
- Device management

**VAETrainer:**

Implements VAE-specific training with:
- Reconstruction loss (MSE)
- KL divergence loss
- Adversarial loss (discriminator after `disc_start` steps)
- Perceptual loss (LPIPS)
- Image sample saving

**Usage:**

```python
from src.ops.config import load_config
from src.models import build_model_from_cfg
from src.trainers import VAETrainer

cfg = load_config('vae')
model = build_model_from_cfg(cfg)
trainer = VAETrainer(model, cfg, device='cuda')
trainer.train(train_loader)
```

### Adding New Models

To add a new model to the system:

1. **Create model file** in `src/models/`:
```python
class MyModel(nn.Module):
    def __init__(self, ...):
        pass

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        # Build model from config
        return cls(...)
```

2. **Register in `src/models/build.py`**:
```python
from .my_model import MyModel
register_model("my_model")(MyModel)
```

3. **Create config file** `configs/my_model.yaml`:
```yaml
extends: base
dataset: celebhq

model:
  name: my_model
  params:
    # model-specific params
```

4. **Create trainer** (optional):
```python
class MyModelTrainer(BaseTrainer):
    def train_step(self, batch):
        # Implement training logic
        pass
```

### Migration Notes

**Phase 1 (Complete):**
- ✅ Fixed all import inconsistencies
- ✅ All code uses correct plural paths

**Phase 2 (Complete):**
- ✅ Config system with inheritance
- ✅ Dataset registry
- ✅ Model registry and builder
- ✅ Trainer abstraction
- ✅ Unified training script

**Legacy Support:**
- Old training scripts (`train_vae.py`, `train_vae_dit.py`) still work
- Can gradually migrate to new system
- New development should use `tools/train.py`

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return types
- Maintain modular, well-organized code structure
- Write clear docstrings for classes and functions
