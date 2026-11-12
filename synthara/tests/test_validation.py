"""
Tests for the validation module (fidelity + privacy).
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.fidelity import FidelityEvaluator
from src.validation.privacy import PrivacyEvaluator


@pytest.fixture
def real_df():
    """Create a sample 'real' dataset."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "amount": np.random.lognormal(10, 2, n).clip(100, 5_000_000),
        "oldbalanceOrg": np.random.lognormal(11, 2, n).clip(0, 10_000_000),
        "newbalanceOrig": np.random.lognormal(11, 2, n).clip(0, 10_000_000),
        "type": np.random.choice(["CASH_OUT", "PAYMENT", "CASH_IN"], size=n, p=[0.4, 0.4, 0.2]),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.99, 0.01]),
    })


@pytest.fixture
def synthetic_good(real_df):
    """Synthetic data that's similar but not identical to real."""
    np.random.seed(99)
    n = 300
    return pd.DataFrame({
        "amount": np.random.lognormal(10, 2.1, n).clip(100, 5_000_000),
        "oldbalanceOrg": np.random.lognormal(11, 2.1, n).clip(0, 10_000_000),
        "newbalanceOrig": np.random.lognormal(11, 2.1, n).clip(0, 10_000_000),
        "type": np.random.choice(["CASH_OUT", "PAYMENT", "CASH_IN"], size=n, p=[0.38, 0.42, 0.2]),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.99, 0.01]),
    })


@pytest.fixture
def synthetic_bad():
    """Synthetic data with totally different distributions."""
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        "amount": np.random.uniform(0, 100, n),  # Completely different
        "oldbalanceOrg": np.random.uniform(0, 50, n),
        "newbalanceOrig": np.random.uniform(0, 50, n),
        "type": np.random.choice(["CASH_OUT", "PAYMENT", "CASH_IN"], size=n, p=[0.1, 0.1, 0.8]),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
    })


class TestFidelityEvaluator:
    def test_evaluate_returns_score(self, real_df, synthetic_good):
        evaluator = FidelityEvaluator()
        report = evaluator.evaluate(real_df, synthetic_good)
        assert "overall_score" in report
        assert 0.0 <= report["overall_score"] <= 1.0

    def test_good_synthetic_scores_higher(self, real_df, synthetic_good, synthetic_bad):
        evaluator = FidelityEvaluator()
        good_report = evaluator.evaluate(real_df, synthetic_good)
        bad_report = evaluator.evaluate(real_df, synthetic_bad)
        assert good_report["overall_score"] > bad_report["overall_score"]

    def test_column_tests_present(self, real_df, synthetic_good):
        evaluator = FidelityEvaluator()
        report = evaluator.evaluate(real_df, synthetic_good)
        assert "column_tests" in report
        assert len(report["column_tests"]) > 0

    def test_correlation_similarity(self, real_df, synthetic_good):
        evaluator = FidelityEvaluator()
        report = evaluator.evaluate(real_df, synthetic_good)
        assert report["correlation_similarity"] is not None


class TestPrivacyEvaluator:
    def test_evaluate_returns_score(self, real_df, synthetic_good):
        evaluator = PrivacyEvaluator()
        report = evaluator.evaluate(real_df, synthetic_good)
        assert "privacy_score" in report
        assert 0.0 <= report["privacy_score"] <= 1.0

    def test_no_exact_matches(self, real_df, synthetic_good):
        evaluator = PrivacyEvaluator()
        report = evaluator.evaluate(real_df, synthetic_good)
        assert report["exact_matches"] == 0

    def test_exact_copy_detected(self, real_df):
        """If synthetic is a copy of real, exact matches should be found."""
        evaluator = PrivacyEvaluator()
        report = evaluator.evaluate(real_df, real_df.copy())
        assert report["exact_matches"] > 0
        assert report["privacy_score"] < 0.8  # Should be penalized
