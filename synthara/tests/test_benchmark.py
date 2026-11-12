"""
Tests for the benchmark module (models + TSTR).
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmark.models import get_model, prepare_features, MODEL_REGISTRY
from src.benchmark.evaluator import evaluate_model, format_comparison_table


@pytest.fixture
def fraud_df():
    """Create a dataset suitable for fraud detection benchmarking."""
    np.random.seed(42)
    n = 500
    is_fraud = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    types = np.random.choice(
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER"],
        size=n, p=[0.35, 0.35, 0.2, 0.1],
    )
    return pd.DataFrame({
        "step": np.random.randint(1, 744, size=n),
        "type": types,
        "amount": np.random.lognormal(10, 2, n).clip(100, 5_000_000),
        "oldbalanceOrg": np.random.lognormal(11, 2, n).clip(0, 10_000_000),
        "newbalanceOrig": np.random.lognormal(11, 2, n).clip(0, 10_000_000),
        "oldbalanceDest": np.random.lognormal(11, 2, n).clip(0, 10_000_000),
        "newbalanceDest": np.random.lognormal(11, 2, n).clip(0, 10_000_000),
        "isFraud": is_fraud,
        "isFlaggedFraud": 0,
    })


class TestModels:
    def test_get_all_models(self):
        for name in MODEL_REGISTRY:
            model = get_model(name)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_model("nonexistent_model")

    def test_prepare_features(self, fraud_df):
        X, y = prepare_features(fraud_df, target_column="isFraud")
        assert len(X) == len(fraud_df)
        assert len(y) == len(fraud_df)
        assert "isFraud" not in X.columns
        assert "nameOrig" not in X.columns  # Should be dropped
        assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]  # All numeric

    def test_model_can_train_and_predict(self, fraud_df):
        X, y = prepare_features(fraud_df)
        model = get_model("random_forest")
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestEvaluator:
    def test_evaluate_model_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.85, 0.15, 0.6])

        metrics = evaluate_model(y_true, y_pred, y_proba, model_name="test")
        assert "roc_auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_format_comparison_table(self):
        results = {
            "Real": {"xgboost": {"roc_auc": 0.95, "f1": 0.8, "precision": 0.75, "recall": 0.85, "accuracy": 0.92}},
            "Synthetic": {"xgboost": {"roc_auc": 0.88, "f1": 0.7, "precision": 0.65, "recall": 0.78, "accuracy": 0.87}},
        }
        table = format_comparison_table(results)
        assert "SYNTHARA BENCHMARK" in table
        assert "xgboost" in table
