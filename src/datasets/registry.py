"""
Dataset Registry System

Central registry for loading dataset configurations from configs/datasets.yaml.
Provides caching and lookup functionality for dataset metadata.
"""

import yaml
import os
from typing import Dict, Any

_DATASET_REGISTRY_CACHE = None


def load_dataset_registry() -> Dict[str, Dict[str, Any]]:
    """Load dataset registry from configs/datasets.yaml.

    Returns:
        Dictionary mapping dataset names to their configurations
    """
    global _DATASET_REGISTRY_CACHE

    if _DATASET_REGISTRY_CACHE is None:
        # Look for registry in project root
        registry_path = os.path.join("configs", "datasets.yaml")

        if not os.path.exists(registry_path):
            raise FileNotFoundError(
                f"Dataset registry not found at {registry_path}. "
                "Please create configs/datasets.yaml"
            )

        with open(registry_path, "r") as f:
            _DATASET_REGISTRY_CACHE = yaml.safe_load(f)

    return _DATASET_REGISTRY_CACHE


def get_dataset_config(name: str) -> Dict[str, Any]:
    """Get dataset configuration by name.

    Args:
        name: Name of the dataset (e.g., 'celebhq', 'imagenet')

    Returns:
        Dictionary with dataset configuration including path, im_size, etc.

    Raises:
        ValueError: If dataset name not found in registry
    """
    registry = load_dataset_registry()

    if name not in registry:
        raise ValueError(
            f"Dataset '{name}' not found in registry. "
            f"Available datasets: {list(registry.keys())}"
        )

    return registry[name]


def list_datasets() -> list:
    """List all available dataset names in the registry.

    Returns:
        List of dataset names
    """
    registry = load_dataset_registry()
    return list(registry.keys())


def clear_cache():
    """Clear the dataset registry cache.

    Useful for testing or when the registry file is modified.
    """
    global _DATASET_REGISTRY_CACHE
    _DATASET_REGISTRY_CACHE = None
