"""
Synthara — Fidelity Metrics
Statistical fidelity evaluation of synthetic data vs real data.
Uses SDMetrics for comprehensive quality scoring and custom KS tests.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class FidelityEvaluator:
    """
    Evaluates how statistically faithful synthetic data is to the original.

    Metrics computed:
    1. Per-column Kolmogorov-Smirnov tests (numeric columns)
    2. Per-column Chi-squared tests (categorical columns)
    3. Correlation matrix similarity (Frobenius norm)
    4. SDMetrics Quality Report (if sdmetrics is available)
    5. Overall fidelity score (0.0 = terrible, 1.0 = perfect)
    """

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        metadata: Optional[object] = None,
    ) -> dict:
        """
        Run full fidelity evaluation.

        Args:
            real_df: Real/source DataFrame.
            synthetic_df: Synthetic DataFrame.
            metadata: Optional SDV SingleTableMetadata object.

        Returns:
            Dictionary with all fidelity metrics.
        """
        logger.info(
            f"Evaluating fidelity: real={real_df.shape}, synthetic={synthetic_df.shape}"
        )

        report = {
            "column_tests": {},
            "correlation_similarity": None,
            "sdmetrics_score": None,
            "overall_score": 0.0,
        }

        # 1. Per-column statistical tests
        scores = []

        # Numeric columns — KS test
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        common_numeric = [c for c in numeric_cols if c in synthetic_df.columns]

        for col in common_numeric:
            real_vals = real_df[col].dropna()
            synth_vals = synthetic_df[col].dropna()

            if len(real_vals) == 0 or len(synth_vals) == 0:
                continue

            ks_stat, ks_pvalue = stats.ks_2samp(real_vals, synth_vals)

            # Score: 1 - KS statistic (lower KS = better match)
            col_score = 1.0 - ks_stat
            scores.append(col_score)

            report["column_tests"][col] = {
                "type": "numeric",
                "test": "kolmogorov_smirnov",
                "ks_statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "score": float(col_score),
                "real_mean": float(real_vals.mean()),
                "synthetic_mean": float(synth_vals.mean()),
                "real_std": float(real_vals.std()),
                "synthetic_std": float(synth_vals.std()),
            }

        # Categorical columns — Chi-squared test
        cat_cols = real_df.select_dtypes(include=["object", "category"]).columns
        common_cat = [c for c in cat_cols if c in synthetic_df.columns]

        for col in common_cat:
            real_counts = real_df[col].value_counts(normalize=True)
            synth_counts = synthetic_df[col].value_counts(normalize=True)

            # Align categories
            all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
            real_freq = np.array([real_counts.get(c, 0) for c in all_cats])
            synth_freq = np.array([synth_counts.get(c, 0) for c in all_cats])

            # Jensen-Shannon divergence (symmetric version of KL divergence)
            m = (real_freq + synth_freq) / 2
            # Avoid log(0)
            real_freq_safe = np.where(real_freq == 0, 1e-10, real_freq)
            synth_freq_safe = np.where(synth_freq == 0, 1e-10, synth_freq)
            m_safe = np.where(m == 0, 1e-10, m)

            js_div = 0.5 * (
                stats.entropy(real_freq_safe, m_safe)
                + stats.entropy(synth_freq_safe, m_safe)
            )

            # Score: 1 - JS divergence (capped at 0)
            col_score = max(0.0, 1.0 - js_div)
            scores.append(col_score)

            report["column_tests"][col] = {
                "type": "categorical",
                "test": "jensen_shannon",
                "js_divergence": float(js_div),
                "score": float(col_score),
                "real_categories": len(real_counts),
                "synthetic_categories": len(synth_counts),
            }

        # 2. Correlation matrix similarity
        if len(common_numeric) >= 2:
            real_corr = real_df[common_numeric].corr().values
            synth_corr = synthetic_df[common_numeric].corr().values

            # Frobenius norm of correlation difference, normalized
            corr_diff = np.linalg.norm(real_corr - synth_corr, "fro")
            max_possible = np.sqrt(2 * len(common_numeric) ** 2)  # Max possible Frobenius norm
            corr_score = max(0.0, 1.0 - corr_diff / max_possible)
            scores.append(corr_score)

            report["correlation_similarity"] = {
                "frobenius_distance": float(corr_diff),
                "score": float(corr_score),
            }

        # 3. SDMetrics Quality Report (optional)
        report["sdmetrics_score"] = self._run_sdmetrics(real_df, synthetic_df, metadata)
        if report["sdmetrics_score"] is not None:
            scores.append(report["sdmetrics_score"]["overall_quality_score"])

        # 4. Overall score
        report["overall_score"] = float(np.mean(scores)) if scores else 0.0
        report["num_columns_evaluated"] = len(report["column_tests"])

        logger.info(f"Fidelity evaluation complete. Overall score: {report['overall_score']:.4f}")
        return report

    def _run_sdmetrics(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        metadata: Optional[object] = None,
    ) -> Optional[dict]:
        """
        Run SDMetrics Quality Report if available.
        """
        try:
            from sdmetrics.reports.single_table import QualityReport
            from sdv.metadata import SingleTableMetadata

            if metadata is None:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(real_df)

            quality_report = QualityReport()
            quality_report.generate(real_df, synthetic_df, metadata.to_dict())

            overall_score = quality_report.get_score()

            return {
                "overall_quality_score": float(overall_score),
            }
        except ImportError:
            logger.warning("sdmetrics not installed, skipping SDMetrics Quality Report")
            return None
        except Exception as e:
            logger.warning(f"SDMetrics evaluation failed: {e}")
            return None


def evaluate_fidelity(
    real_path: str,
    synthetic_path: str,
    nrows: Optional[int] = None,
) -> dict:
    """
    Convenience function: load CSVs and evaluate fidelity.

    Args:
        real_path: Path to real data CSV.
        synthetic_path: Path to synthetic data CSV.
        nrows: Optional max rows to load.

    Returns:
        Fidelity report dictionary.
    """
    real_df = pd.read_csv(real_path, nrows=nrows)
    synthetic_df = pd.read_csv(synthetic_path, nrows=nrows)

    evaluator = FidelityEvaluator()
    return evaluator.evaluate(real_df, synthetic_df)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python -m src.validation.fidelity <real_csv> <synthetic_csv>")
        sys.exit(1)

    report = evaluate_fidelity(sys.argv[1], sys.argv[2])
    print(json.dumps(report, indent=2))
