"""
Synthara — Benchmark Evaluator
Computes and formats ML evaluation metrics for the TSTR benchmark.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "unknown",
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for ROC-AUC).
        model_name: Name of the model for the report.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC (requires probabilities)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

    # Class distribution in test set
    unique, counts = np.unique(y_true, return_counts=True)
    metrics["test_class_distribution"] = {
        str(k): int(v) for k, v in zip(unique, counts)
    }

    return metrics


def format_comparison_table(results: dict) -> str:
    """
    Format benchmark results as a readable comparison table.

    Args:
        results: Dict of {scenario_name: {model_name: metrics_dict}}.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("=" * 90)
    lines.append("SYNTHARA BENCHMARK REPORT — TSTR (Train Synthetic, Test Real)")
    lines.append("=" * 90)

    metrics_to_show = ["roc_auc", "f1", "precision", "recall", "accuracy"]

    for scenario, models in results.items():
        lines.append(f"\n📊 Scenario: {scenario}")
        lines.append("-" * 70)

        header = f"{'Model':<25}" + "".join(f"{m.upper():<12}" for m in metrics_to_show)
        lines.append(header)
        lines.append("-" * 70)

        for model_name, metrics in models.items():
            row = f"{model_name:<25}"
            for m in metrics_to_show:
                val = metrics.get(m, 0)
                row += f"{val:<12.4f}"
            lines.append(row)

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)
