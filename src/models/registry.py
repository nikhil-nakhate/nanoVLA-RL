"""
Model Registry System

Provides decorator-based model registration for config-driven model building.
"""

from typing import Dict, Type, Any


_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register a model class.

    Usage:
        @register_model("vae")
        class VAE(nn.Module):
            @classmethod
            def from_config(cls, cfg: Dict[str, Any]):
                # Build model from config
                return cls(...)

    Args:
        name: Unique name for the model (used in config files)

    Returns:
        Decorator function that registers the class

    Raises:
        ValueError: If model name already registered
    """
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' already registered. "
                f"Existing class: {_MODEL_REGISTRY[name].__name__}"
            )

        # Verify the class has from_config method
        if not hasattr(cls, "from_config"):
            raise AttributeError(
                f"Model class {cls.__name__} must implement 'from_config' classmethod. "
                "This method should accept a config dict and return an instance of the model."
            )

        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type:
    """Retrieve a registered model class by name.

    Args:
        name: Name of the registered model

    Returns:
        The model class

    Raises:
        KeyError: If model name not found in registry
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    return _MODEL_REGISTRY[name]


def list_models() -> list:
    """List all registered model names.

    Returns:
        List of model names
    """
    return list(_MODEL_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model name is registered.

    Args:
        name: Model name to check

    Returns:
        True if model is registered, False otherwise
    """
    return name in _MODEL_REGISTRY


def clear_registry():
    """Clear all registered models.

    Useful for testing. Should not be called in production code.
    """
    global _MODEL_REGISTRY
    _MODEL_REGISTRY = {}
