"""
Configuration loading system with hierarchical YAML support.

Supports config inheritance via 'extends' keyword and deep merging.
Adapted from open-slm-agents for nanoVLA-RL's flat config structure.
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml


CONFIGS_ROOT = os.path.abspath("configs")


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(
    path_or_name: str,
    search_dir: str,
) -> str:
    """Resolve a config reference to an absolute path.

    Searches for configs in the following order:
    1. Absolute path if provided
    2. Relative path from current directory
    3. In the search_dir with .yaml extension
    4. In the search_dir with .yml extension
    5. In the search_dir without modification

    Args:
        path_or_name: Path or name of config file
        search_dir: Directory to search for configs

    Returns:
        Absolute path to resolved config file

    Raises:
        FileNotFoundError: If config file cannot be found
    """
    candidates = []

    # If absolute path, try it directly
    if os.path.isabs(path_or_name):
        candidates.append(path_or_name)
    else:
        # If relative path exists from current directory
        if os.path.exists(path_or_name):
            candidates.append(os.path.abspath(path_or_name))

        # Check if name has extension
        name = path_or_name
        name_has_ext = os.path.splitext(name)[1] in {".yaml", ".yml"}

        # Search in search_dir
        if not name_has_ext:
            candidates.append(os.path.join(search_dir, f"{name}.yaml"))
            candidates.append(os.path.join(search_dir, f"{name}.yml"))
        candidates.append(os.path.join(search_dir, name))

    # Try each candidate
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)

    raise FileNotFoundError(
        f"Config file not found: {path_or_name}\n"
        f"Searched in: {search_dir}"
    )


def load_config(
    config_path_or_name: str,
    search_dir: str = "configs",
) -> Dict[str, Any]:
    """Load a YAML config with hierarchical inheritance support.

    Supports hierarchical configs via an 'extends' key that references another
    YAML file. Performs a deep merge where the child overrides the parent.

    Examples:
        >>> # Load from configs/vae.yaml
        >>> cfg = load_config("vae")

        >>> # Load with absolute path
        >>> cfg = load_config("/path/to/custom.yaml")

        >>> # Load from custom directory
        >>> cfg = load_config("model", search_dir="/custom/configs")

        >>> # Config inheritance example (vae.yaml):
        >>> # extends: base
        >>> # model:
        >>> #   name: vae
        >>> #   params:
        >>> #     z_channels: 4

    Args:
        config_path_or_name: Path or name of config file (extension optional)
        search_dir: Directory to search for config files (default: "configs")

    Returns:
        Configuration dictionary with deep-merged parent configs

    Raises:
        FileNotFoundError: If config file cannot be found
        yaml.YAMLError: If YAML parsing fails
    """
    search_dir_abs = os.path.abspath(search_dir)
    resolved_path = _resolve_path(config_path_or_name, search_dir_abs)

    with open(resolved_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Handle hierarchical base config via 'extends'
    extends = cfg.pop("extends", None)
    if extends:
        # Load parent config from same search directory
        parent_cfg = load_config(extends, search_dir=search_dir_abs)
        # Deep merge: child overrides parent
        cfg = _deep_update(parent_cfg, cfg)

    return cfg


def save_config(cfg: Dict[str, Any], output_path: str):
    """Save a configuration dictionary to a YAML file.

    Args:
        cfg: Configuration dictionary to save
        output_path: Path to save the YAML file

    Raises:
        IOError: If file cannot be written
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
