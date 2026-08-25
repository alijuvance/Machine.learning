"""
Synthara — ML Models for Benchmarking
Defines the ML models used in the TSTR (Train Synthetic, Test Real) benchmark.
Currently supports fraud detection; credit scoring coming in Phase 2.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def get_xgboost_model(config: Optional[dict] = None):
    """
    Create an XGBoost classifier configured for imbalanced fraud detection.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed, falling back to RandomForest")
        return get_random_forest_model(config)

    default_config = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    }

    if config:
        default_config.update(config)

    # Handle auto scale_pos_weight
    if default_config.get("scale_pos_weight") == "auto":
        del default_config["scale_pos_weight"]

    return XGBClassifier(**default_config)


def get_random_forest_model(config: Optional[dict] = None):
    """Create a RandomForest classifier with balanced class weights."""
    default_config = {
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    if config:
        default_config.update(config)
    return RandomForestClassifier(**default_config)


def get_logistic_regression_model(config: Optional[dict] = None):
    """Create a LogisticRegression with balanced class weights."""
    default_config = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": 42,
    }
    if config:
        default_config.update(config)
    return LogisticRegression(**default_config)


MODEL_REGISTRY = {
    "xgboost": get_xgboost_model,
    "random_forest": get_random_forest_model,
    "logistic_regression": get_logistic_regression_model,
}


def get_model(name: str, config: Optional[dict] = None):
    """
    Get a model by name from the registry.

    Args:
        name: Model name (xgboost, random_forest, logistic_regression).
        config: Optional model-specific configuration.

    Returns:
        Configured sklearn-compatible model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    drop_columns: Optional[list] = None,
    encode_columns: Optional[list] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for ML training.

    Handles:
    - Dropping irrelevant columns (IDs, flags)
    - Label encoding categorical columns
    - Handling missing values

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        drop_columns: Columns to drop.
        encode_columns: Categorical columns to label-encode.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    if drop_columns is None:
        drop_columns = ["nameOrig", "nameDest", "isFlaggedFraud"]
    if encode_columns is None:
        encode_columns = ["type"]

    df = df.copy()

    # Extract target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    y = df[target_column].astype(int)

    # Drop target and irrelevant columns
    cols_to_drop = [c for c in [target_column] + drop_columns if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Encode categorical columns
    for col in encode_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Drop remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.info(f"Dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Fill NaN
    X = X.fillna(0)

    return X, y
