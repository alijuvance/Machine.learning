"""
Tests for the data analysis profiler module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_analysis.profiler import DataProfiler, DatasetProfile


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame mimicking PaySim structure."""
    np.random.seed(42)
    n = 500
    types = np.random.choice(
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n, p=[0.35, 0.34, 0.22, 0.08, 0.01],
    )
    return pd.DataFrame({
        "step": np.random.randint(1, 744, size=n),
        "type": types,
        "amount": np.random.lognormal(mean=10, sigma=2, size=n).clip(100, 5_000_000),
        "oldbalanceOrg": np.random.lognormal(mean=11, sigma=2, size=n).clip(0, 10_000_000),
        "newbalanceOrig": np.random.lognormal(mean=11, sigma=2, size=n).clip(0, 10_000_000),
        "oldbalanceDest": np.random.lognormal(mean=11, sigma=2, size=n).clip(0, 10_000_000),
        "newbalanceDest": np.random.lognormal(mean=11, sigma=2, size=n).clip(0, 10_000_000),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.99, 0.01]),
        "isFlaggedFraud": 0,
    })


class TestDataProfiler:
    def test_profile_returns_profile_object(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df, name="test")
        assert isinstance(result, DatasetProfile)
        assert result.name == "test"

    def test_profile_shape(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df)
        assert result.shape == sample_df.shape

    def test_numeric_stats_computed(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df)
        assert result.numeric_stats is not None
        assert "amount" in result.numeric_stats.index
        assert "mean" in result.numeric_stats.columns

    def test_categorical_stats_computed(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df)
        assert "type" in result.categorical_stats
        assert "unique_values" in result.categorical_stats["type"]
        assert result.categorical_stats["type"]["unique_values"] == 5

    def test_correlations_computed(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df)
        assert result.correlations is not None
        assert result.correlations.shape[0] > 0

    def test_outlier_detection(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df)
        assert isinstance(result.outlier_counts, dict)

    def test_summary_returns_dict(self, sample_df):
        profiler = DataProfiler()
        result = profiler.profile(sample_df)
        summary = result.summary()
        assert isinstance(summary, dict)
        assert "rows" in summary
        assert "columns" in summary
        assert summary["rows"] == 500

    def test_compare_profiles(self, sample_df):
        profiler = DataProfiler()
        profile1 = profiler.profile(sample_df, name="real")
        profile2 = profiler.profile(sample_df, name="synth")  # Same data = perfect match

        comparison = profiler.compare_profiles(profile1, profile2)
        assert "overall_score" in comparison
        assert comparison["overall_score"] >= 0.9  # Same data should score high
