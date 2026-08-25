"""
Synthara — Statistical Profiler
Automated statistical analysis of source datasets.
Produces a comprehensive profile: distributions, correlations, missing values,
outliers, and summary statistics for every column.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DatasetProfile:
    """Container for all profiling results of a dataset."""

    def __init__(self, name: str):
        self.name = name
        self.shape: tuple = (0, 0)
        self.dtypes: dict = {}
        self.missing: dict = {}
        self.numeric_stats: Optional[pd.DataFrame] = None
        self.categorical_stats: dict = {}
        self.correlations: Optional[pd.DataFrame] = None
        self.outlier_counts: dict = {}
        self.distribution_fits: dict = {}

    def summary(self) -> dict:
        """Return a JSON-serializable summary of the profile."""
        return {
            "name": self.name,
            "rows": self.shape[0],
            "columns": self.shape[1],
            "numeric_columns": len([
                d for d in self.dtypes.values()
                if d in ("int64", "float64", "int32", "float32")
            ]),
            "categorical_columns": len([
                d for d in self.dtypes.values()
                if d in ("object", "category", "bool")
            ]),
            "total_missing": sum(self.missing.values()),
            "total_outliers": sum(self.outlier_counts.values()),
        }


class DataProfiler:
    """
    Performs automated statistical profiling of tabular datasets.

    Usage:
        profiler = DataProfiler()
        profile = profiler.profile(df, name="PaySim")
        print(profile.summary())
    """

    def __init__(self, outlier_method: str = "iqr", outlier_threshold: float = 1.5):
        """
        Args:
            outlier_method: Method for outlier detection. "iqr" (default) or "zscore".
            outlier_threshold: Threshold for outlier detection.
                IQR: multiplier (default 1.5). Z-score: number of std devs (default 3.0).
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

    def profile(self, df: pd.DataFrame, name: str = "dataset") -> DatasetProfile:
        """
        Run full profiling pipeline on a DataFrame.

        Args:
            df: Input DataFrame to profile.
            name: Human-readable name for this dataset.

        Returns:
            DatasetProfile with all analysis results.
        """
        logger.info(f"Profiling dataset '{name}': {df.shape[0]} rows × {df.shape[1]} columns")
        result = DatasetProfile(name)
        result.shape = df.shape
        result.dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Missing values analysis
        result.missing = self._analyze_missing(df)

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            result.numeric_stats = self._numeric_statistics(df[numeric_cols])
            result.correlations = df[numeric_cols].corr()
            result.outlier_counts = self._detect_outliers(df[numeric_cols])
            result.distribution_fits = self._fit_distributions(df[numeric_cols])

        # Categorical columns analysis
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if cat_cols:
            result.categorical_stats = self._categorical_statistics(df[cat_cols])

        logger.info(f"Profiling complete: {result.summary()}")
        return result

    def _analyze_missing(self, df: pd.DataFrame) -> dict:
        """Count missing values per column."""
        missing = df.isnull().sum()
        return {col: int(count) for col, count in missing.items() if count > 0}

    def _numeric_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute detailed statistics for numeric columns."""
        stats_df = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        stats_df["skewness"] = df.skew()
        stats_df["kurtosis"] = df.kurtosis()
        stats_df["median"] = df.median()
        stats_df["iqr"] = stats_df["75%"] - stats_df["25%"]
        stats_df["cv"] = (stats_df["std"] / stats_df["mean"]).replace(
            [np.inf, -np.inf], np.nan
        )
        return stats_df

    def _categorical_statistics(self, df: pd.DataFrame) -> dict:
        """Compute value counts and entropy for categorical columns."""
        result = {}
        for col in df.columns:
            value_counts = df[col].value_counts()
            probabilities = value_counts / len(df)
            entropy = stats.entropy(probabilities)
            result[col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": value_counts.head(10).to_dict(),
                "entropy": float(entropy),
                "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            }
        return result

    def _detect_outliers(self, df: pd.DataFrame) -> dict:
        """Detect outliers using IQR or Z-score method."""
        outlier_counts = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            if self.outlier_method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
                n_outliers = int(((series < lower) | (series > upper)).sum())
            else:  # zscore
                z_scores = np.abs(stats.zscore(series))
                n_outliers = int((z_scores > self.outlier_threshold).sum())

            if n_outliers > 0:
                outlier_counts[col] = n_outliers

        return outlier_counts

    def _fit_distributions(self, df: pd.DataFrame) -> dict:
        """
        Fit common distributions to numeric columns and select best fit
        using Kolmogorov-Smirnov test.
        """
        distributions_to_try = ["norm", "lognorm", "expon", "gamma", "beta"]
        results = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 50:
                continue

            best_dist = None
            best_pvalue = 0.0

            for dist_name in distributions_to_try:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(series)
                    ks_stat, p_value = stats.kstest(series, dist_name, args=params)
                    if p_value > best_pvalue:
                        best_pvalue = p_value
                        best_dist = {
                            "distribution": dist_name,
                            "params": [float(p) for p in params],
                            "ks_statistic": float(ks_stat),
                            "p_value": float(p_value),
                        }
                except Exception:
                    continue

            if best_dist:
                results[col] = best_dist

        return results

    def compare_profiles(
        self, real_profile: DatasetProfile, synthetic_profile: DatasetProfile
    ) -> dict:
        """
        Compare two profiles (real vs synthetic) and return divergence metrics.

        Args:
            real_profile: Profile of the real/source dataset.
            synthetic_profile: Profile of the synthetic dataset.

        Returns:
            Dictionary with comparison metrics per column.
        """
        comparison = {"column_comparisons": {}, "overall_score": 0.0}

        if real_profile.numeric_stats is not None and synthetic_profile.numeric_stats is not None:
            common_cols = set(real_profile.numeric_stats.index) & set(
                synthetic_profile.numeric_stats.index
            )
            scores = []
            for col in common_cols:
                real_stats = real_profile.numeric_stats.loc[col]
                synth_stats = synthetic_profile.numeric_stats.loc[col]

                # Compare means and stds
                mean_diff = abs(real_stats["mean"] - synth_stats["mean"])
                std_diff = abs(real_stats["std"] - synth_stats["std"])

                # Normalize by real values to get relative differences
                mean_rel = (
                    mean_diff / abs(real_stats["mean"])
                    if real_stats["mean"] != 0
                    else mean_diff
                )
                std_rel = (
                    std_diff / abs(real_stats["std"])
                    if real_stats["std"] != 0
                    else std_diff
                )

                col_score = max(0.0, 1.0 - (mean_rel + std_rel) / 2)
                scores.append(col_score)

                comparison["column_comparisons"][col] = {
                    "mean_real": float(real_stats["mean"]),
                    "mean_synthetic": float(synth_stats["mean"]),
                    "std_real": float(real_stats["std"]),
                    "std_synthetic": float(synth_stats["std"]),
                    "similarity_score": float(col_score),
                }

            if scores:
                comparison["overall_score"] = float(np.mean(scores))

        return comparison


def load_and_profile(filepath: str, name: Optional[str] = None, nrows: Optional[int] = None) -> DatasetProfile:
    """
    Convenience function: load a CSV and profile it.

    Args:
        filepath: Path to CSV file.
        name: Optional name for the dataset (defaults to filename).
        nrows: Optional max number of rows to load (for large files).

    Returns:
        DatasetProfile with analysis results.
    """
    path = Path(filepath)
    if name is None:
        name = path.stem

    logger.info(f"Loading {filepath}...")
    df = pd.read_csv(filepath, nrows=nrows)
    logger.info(f"Loaded {len(df)} rows")

    profiler = DataProfiler()
    return profiler.profile(df, name=name)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.data_analysis.profiler <csv_path> [--nrows N]")
        sys.exit(1)

    csv_path = sys.argv[1]
    nrows = None
    if "--nrows" in sys.argv:
        nrows = int(sys.argv[sys.argv.index("--nrows") + 1])

    profile = load_and_profile(csv_path, nrows=nrows)
    print(json.dumps(profile.summary(), indent=2))
