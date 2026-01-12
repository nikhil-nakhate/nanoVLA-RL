"""
Model registry and building utilities

Provides decorator-based model registration and config-driven model building.
"""

from .registry import (
    register_model,
    get_model_class,
    list_models,
    is_registered,
    clear_registry
)
from .build import build_model_from_cfg, register_models

# Auto-register all models
register_models()

__all__ = [
    'register_model',
    'get_model_class',
    'list_models',
    'is_registered',
    'clear_registry',
    'build_model_from_cfg',
    'register_models'
]
