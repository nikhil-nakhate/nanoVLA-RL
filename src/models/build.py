"""
Model builder for config-driven model instantiation

Provides functions to build models from configuration dictionaries using the registry pattern.
"""

from typing import Dict, Any
from .registry import get_model_class


def build_model_from_cfg(cfg: Dict[str, Any]):
    """Build a model from configuration dictionary.

    The config must include a 'model.name' field that references a registered model.
    The model class must implement a 'from_config' classmethod.

    Config structure:
        model:
            name: <registered_model_name>  # e.g., "vae", "dit", "paligemma"
            params: {...}                  # Model-specific parameters
            weights: <optional_path>       # Optional pretrained weights

    Example:
        >>> from src.ops.config import load_config
        >>> from src.models.build import build_model_from_cfg
        >>>
        >>> cfg = load_config("vae")
        >>> model = build_model_from_cfg(cfg)

    Args:
        cfg: Configuration dictionary with model specifications

    Returns:
        Instantiated model

    Raises:
        KeyError: If model.name not found in config or registry
        AttributeError: If model class doesn't implement from_config
    """
    model_cfg = cfg.get("model")
    if not model_cfg:
        raise KeyError("Config must include 'model' section")

    model_name = model_cfg.get("name")
    if not model_name:
        raise KeyError("Config must include 'model.name' field")

    # Get registered model class
    ModelCls = get_model_class(model_name)

    # Verify from_config exists
    if not hasattr(ModelCls, "from_config"):
        raise AttributeError(
            f"Model class '{ModelCls.__name__}' must implement 'from_config' classmethod"
        )

    # Build model using from_config
    model = ModelCls.from_config(cfg)

    return model


def register_models():
    """Register all available models.

    This function imports and registers all model classes.
    Should be called before using build_model_from_cfg.
    """
    from .registry import register_model

    # Import and register VAE
    try:
        from .vae import VAE
        if not hasattr(VAE, '_registered'):
            register_model("vae")(VAE)
            VAE._registered = True
    except ImportError:
        pass

    # Import and register DiT
    try:
        from .dit import DIT
        if not hasattr(DIT, '_registered'):
            register_model("dit")(DIT)
            DIT._registered = True
    except ImportError:
        pass

    # Import and register PaliGemma (when available)
    try:
        from .paligemma import PaliGemmaForConditionalGeneration
        if not hasattr(PaliGemmaForConditionalGeneration, '_registered'):
            register_model("paligemma")(PaliGemmaForConditionalGeneration)
            PaliGemmaForConditionalGeneration._registered = True
    except ImportError:
        pass


# Auto-register models when module is imported
register_models()
