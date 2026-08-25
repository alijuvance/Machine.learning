"""
Synthara — Generation Config
Loads and validates YAML configuration for generation models.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "generation_config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load generation configuration from YAML file.

    Args:
        config_path: Path to YAML config. Defaults to configs/generation_config.yaml.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(f"Config file not found at {path}, using defaults")
        return _default_config()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded generation config from {path}")
    return config


def _default_config() -> dict:
    """Return default configuration when no YAML file is available."""
    return {
        "models": {
            "ctgan": {
                "epochs": 500,
                "batch_size": 500,
                "verbose": True,
                "enforce_rounding": True,
            },
            "gaussian_copula": {
                "enforce_min_max_values": True,
                "default_distribution": "beta",
            },
            "tvae": {
                "epochs": 500,
                "batch_size": 500,
                "verbose": True,
            },
        },
        "generation": {
            "default_model": "ctgan",
            "default_num_rows": 10000,
            "random_seed": 42,
            "preserve_fraud_ratio": True,
            "fraud_ratio_override": None,
        },
        "constraints": {
            "mobile_money": {
                "min_transaction_amount": 10,
                "max_transaction_amount": 5_000_000,
                "balance_consistency": True,
            }
        },
    }


def get_model_params(config: dict, model_name: str) -> dict:
    """
    Extract model-specific parameters from config.

    Args:
        config: Full configuration dictionary.
        model_name: One of "ctgan", "gaussian_copula", "tvae".

    Returns:
        Model parameters dictionary.
    """
    models_config = config.get("models", {})
    params = models_config.get(model_name, {})
    return params


def get_generation_params(config: dict) -> dict:
    """Extract generation-level parameters."""
    return config.get("generation", {})


def get_constraint_params(config: dict) -> dict:
    """Extract constraint parameters."""
    return config.get("constraints", {})
