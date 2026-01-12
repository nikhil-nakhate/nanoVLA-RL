# nanoVLA-RL

A modular, config-driven framework for training vision and generative models with reinforcement learning capabilities.

## 🎯 Project Overview

nanoVLA-RL is a PyTorch-based framework that combines:
- **Variational Autoencoders (VAE)** for image compression and latent encoding
- **Diffusion Transformers (DiT)** for high-quality image generation in latent space
- **Vision-Language Models (PaliGemma)** for multimodal understanding
- **Config-driven architecture** for easy experimentation and model swapping

The project uses a two-stage generative modeling approach:
1. **Stage 1**: Train VAE to compress images into efficient latent representations
2. **Stage 2**: Train DiT in latent space for diffusion-based generation

## ✨ Features

### Current Implementation

#### Phase 1: Core Infrastructure ✅
- ✅ **VAE Training** - Full implementation with discriminator and perceptual loss (LPIPS)
- ✅ **DiT Training** - Transformer-based diffusion in latent space
- ✅ **PaliGemma Inference** - Vision-language model with SigLIP + Gemma
- ✅ **Import Path Fixes** - Resolved all singular/plural inconsistencies
- ✅ **Device Support** - CUDA, MPS (Apple Silicon), and CPU

#### Phase 2: Config-Driven Architecture ✅
- ✅ **Hierarchical Configs** - YAML configs with inheritance (`extends` keyword)
- ✅ **Dataset Registry** - Central registry mapping dataset names to paths
- ✅ **Model Registry** - Decorator-based model registration with `from_config()`
- ✅ **Trainer Abstraction** - `BaseTrainer` with model-specific subclasses
- ✅ **Unified Training Script** - Single entry point for all models
- ✅ **Atomic Checkpointing** - Safe checkpoint saving with atomic file operations

### Future Extensions

#### Phase 3: Policy Learning 🚧
- 🔲 **Pi0.5 Integration** - Add policy learning capabilities
- 🔲 **RL Trainer** - Reinforcement learning trainer class
- 🔲 **Reward Models** - Configurable reward functions
- 🔲 **Policy Gradient Methods** - PPO, REINFORCE, etc.

#### Phase 4: Simulation & Robotics 🚧
- 🔲 **Simulator Support** - Integration with robotics simulators
- 🔲 **Environment Registry** - Similar to dataset registry for environments
- 🔲 **Vision-Action Models** - VLA architectures for robotic control
- 🔲 **Multi-task Learning** - Joint training across multiple tasks

#### Phase 5: Advanced Features 🚧
- 🔲 **Distributed Training** - DDP support for multi-GPU training
- 🔲 **Experiment Tracking** - W&B/TensorBoard integration
- 🔲 **Mixed Precision Training** - FP16/BF16 for faster training
- 🔲 **Model Quantization** - Post-training quantization for deployment
- 🔲 **DiTTrainer** - Complete DiT training implementation with scheduler
- 🔲 **Inference Scripts** - Unified inference interface matching training

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd nanoVLA-RL
```

2. **Create and activate conda environment:**
```bash
conda create -n llm_agents python=3.10
conda activate llm_agents
```

3. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
```

### Dataset Setup

Download CelebA-HQ dataset and organize as:
```
data/
└── CelebAMask-HQ/
    └── CelebA-HQ-img/
        ├── 0.png
        ├── 1.png
        └── ...
```

Update `configs/datasets.yaml` with your data path.

## 🚀 Quick Start

### Training

**Train VAE (Stage 1):**
```bash
# Using new unified training script
python tools/train.py --config configs/vae.yaml --device cuda

# Specify custom device
python tools/train.py --config configs/vae.yaml --device mps

# Resume from checkpoint
python tools/train.py --config configs/vae.yaml --resume outputs/checkpoint.pt
```

**Train DiT (Stage 2):**
```bash
# Coming soon - DiT training with new system
python tools/train.py --config configs/dit.yaml

# Legacy script (currently working)
python tools/train_vae_dit.py --config configs/celebhq.yaml
```

### Inference

**VAE Reconstruction:**
```bash
python tools/infer_vae.py --config configs/celebhq.yaml
```

**DiT Sampling:**
```bash
python tools/sample_vae_dit.py --config configs/celebhq.yaml
```

**PaliGemma Vision-Language:**
```bash
python tools/infer_paligemma.py \
  --model_path weights/paligemma-3b-pt-224 \
  --prompt "describe this image" \
  --image_file_path test.jpg \
  --max_tokens_to_generate 100
```

## 🏗️ Architecture

### Directory Structure

```
nanoVLA-RL/
├── configs/                    # Configuration files
│   ├── datasets.yaml          # Dataset registry
│   ├── base.yaml              # Base config
│   └── vae.yaml               # Model-specific configs
│
├── src/                       # New config-driven source
│   ├── ops/
│   │   └── config.py         # Config loading with inheritance
│   ├── models/
│   │   ├── registry.py       # Model registration
│   │   ├── build.py          # Model builder
│   │   ├── vae.py            # VAE with from_config()
│   │   └── dit.py            # DiT with from_config()
│   ├── trainers/
│   │   ├── base.py           # BaseTrainer abstract class
│   │   └── vae_trainer.py    # VAETrainer implementation
│   └── datasets/
│       └── registry.py       # Dataset registry loader
│
├── models/                    # Original model implementations
│   ├── vae.py
│   ├── transformer.py        # DiT
│   ├── gemma.py             # Gemma LLM
│   ├── siglip.py            # SigLIP vision encoder
│   ├── blocks.py            # VAE building blocks
│   ├── discriminator.py     # GAN discriminator
│   └── lpips.py             # Perceptual loss
│
├── datasets/
│   └── celeb_dataset.py     # CelebA-HQ dataset loader
│
├── scheduler/
│   └── linear_scheduler.py  # DDPM noise scheduler
│
├── tools/
│   ├── train.py             # NEW: Unified training script
│   ├── train_vae.py         # Legacy VAE training
│   ├── train_vae_dit.py     # Legacy DiT training
│   ├── infer_vae.py         # VAE inference
│   ├── sample_vae_dit.py    # DiT sampling
│   └── infer_paligemma.py   # PaliGemma inference
│
└── utils/
    └── diffusion_utils.py   # Diffusion utilities
```

### Key Components

#### Config System
- **Hierarchical inheritance** via `extends` keyword
- **Deep merging** of parent and child configs
- **Dataset registry** for centralized dataset management
- **Path resolution** with flexible search

#### Model Registry
- **Decorator-based registration**: `@register_model("vae")`
- **Config-driven building**: `build_model_from_cfg(cfg)`
- **Automatic registration** via `src/models/build.py`

#### Trainer System
- **BaseTrainer**: Abstract class with common training loop
- **Model-specific trainers**: VAETrainer, DITTrainer (coming soon)
- **Automatic checkpointing** with atomic file operations
- **Device management** with auto-detection

## 📝 Configuration

### Example Config Structure

```yaml
# configs/vae.yaml
extends: base  # Inherit from base config

dataset: celebhq  # Reference to configs/datasets.yaml

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
  weights: null  # Optional pretrained weights

train:
  epochs: 3
  lr: 0.00001
  batch_size: 4
  disc_start: 7500
  disc_weight: 0.5
  perceptual_weight: 1.0
  kl_weight: 0.000005
  output_dir: 'outputs/celebhq_vae'
```

### Dataset Registry

```yaml
# configs/datasets.yaml
celebhq:
  path: 'data/CelebAMask-HQ'
  im_size: 128
  im_channels: 3
  type: image_folder

imagenet:
  path: 'data/imagenet'
  im_size: 256
  im_channels: 3
  type: image_folder
```

## 🔧 Adding New Models

1. **Create model in `src/models/`:**
```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Model initialization

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MyModel":
        """Build model from config."""
        # Extract params from config
        return cls(...)
```

2. **Register in `src/models/build.py`:**
```python
from .my_model import MyModel
register_model("my_model")(MyModel)
```

3. **Create config file:**
```yaml
# configs/my_model.yaml
extends: base
dataset: celebhq

model:
  name: my_model
  params:
    # model-specific parameters
```

4. **Create trainer (optional):**
```python
class MyModelTrainer(BaseTrainer):
    def train_step(self, batch):
        # Training logic
        pass
```

5. **Train:**
```bash
python tools/train.py --config configs/my_model.yaml
```

## 🎓 Model Details

### VAE (Variational Autoencoder)
- **Architecture**: Encoder-decoder with latent sampling
- **Training losses**:
  - Reconstruction (MSE)
  - KL divergence
  - Perceptual (LPIPS)
  - Adversarial (GAN discriminator)
- **Latent space**: Configurable channels (default: 4)
- **Applications**: Image compression, generative modeling

### DiT (Diffusion Transformer)
- **Architecture**: Vision transformer with adaptive layer normalization
- **Input**: Noisy latents + timestep
- **Output**: Predicted noise
- **Training**: DDPM-style diffusion in latent space
- **Sampling**: Iterative denoising from random noise

### PaliGemma
- **Vision encoder**: SigLIP (patch-based ViT)
- **Language model**: Gemma (decoder-only transformer)
- **Features**:
  - Grouped Query Attention (GQA)
  - RoPE (Rotary Position Embeddings)
  - KV caching for efficient generation
- **Applications**: Image captioning, VQA, visual reasoning

## 📊 Training Tips

### VAE Training
- Start with low learning rate (`1e-5`)
- Enable discriminator after `disc_start` steps (default: 7500)
- Monitor reconstruction quality via saved samples
- Use perceptual loss for better visual quality

### DiT Training
- Pre-train VAE first for quality latents
- Use larger batch sizes when possible
- Adjust number of diffusion timesteps (default: 1000)
- Sample periodically to monitor generation quality

### Device Selection
- **CUDA**: Best performance for NVIDIA GPUs
- **MPS**: For Apple Silicon (M1/M2/M3)
- **CPU**: Fallback, much slower

## 🔬 Research Directions

### Immediate Next Steps
1. **Complete DITTrainer** implementation with scheduler integration
2. **Add unified inference script** matching training interface
3. **Implement model evaluation metrics** (FID, IS, LPIPS)

### Policy Learning (Pi0.5)
1. Add RL trainer with PPO/REINFORCE
2. Design reward model interface
3. Integrate with simulators
4. Multi-task policy learning

### Simulation & Robotics
1. Simulator registry (similar to datasets)
2. Environment wrappers for standardized interface
3. Vision-Language-Action (VLA) models
4. Imitation learning from demonstrations

### Advanced Features
1. Distributed training with DDP
2. Mixed precision (FP16/BF16)
3. Experiment tracking (W&B, TensorBoard)
4. Model compression and quantization
5. ONNX export for deployment

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the existing code style
4. Test your changes thoroughly
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write clear docstrings for classes and methods
- Maintain modular, well-organized code structure

## 📚 Documentation

- **CLAUDE.md**: Detailed guidance for Claude Code on working with this repository
- **Configs**: YAML files in `configs/` with inline comments
- **Code**: Docstrings in source files

## 🐛 Known Issues

- DiT unified trainer not yet implemented (use legacy script)
- No distributed training support yet
- Limited dataset types (only image folders)
- No experiment tracking integration



---

**Status**: Active development 🚀
**Last Updated**: January 2026
**Phase**: 2/5 Complete (Config-driven architecture implemented)
