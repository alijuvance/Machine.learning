"""
Synthara — Privacy Evaluator
Checks that synthetic data does not allow re-identification of individuals
from the original dataset.

Key metric: DCR (Distance to Closest Record) — measures how far each
synthetic record is from the nearest real record. If too close, the
synthetic data may be leaking real information.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PrivacyEvaluator:
    """
    Evaluates privacy protection of synthetic data by measuring
    the distance between synthetic and real records.

    A good synthetic dataset should:
    1. Have no exact copies of real records
    2. Have a high average DCR (Distance to Closest Record)
    3. Not have any synthetic record suspiciously close to a real one
    """

    def __init__(
        self,
        dcr_threshold: float = 0.05,
        sample_size: int = 5000,
    ):
        """
        Args:
            dcr_threshold: Minimum acceptable normalized DCR.
                Records closer than this are flagged as privacy risks.
            sample_size: Max number of records to use for distance computation
                (full pairwise is O(n²), so we sample for large datasets).
        """
        self.dcr_threshold = dcr_threshold
        self.sample_size = sample_size

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
    ) -> dict:
        """
        Run full privacy evaluation.

        Args:
            real_df: Real/source DataFrame.
            synthetic_df: Synthetic DataFrame.

        Returns:
            Dictionary with privacy metrics.
        """
        logger.info("Evaluating privacy protection...")

        report = {
            "exact_matches": 0,
            "dcr_stats": {},
            "records_below_threshold": 0,
            "privacy_score": 0.0,
        }

        # Use only common columns
        common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        real = real_df[common_cols].copy()
        synth = synthetic_df[common_cols].copy()

        # 1. Check for exact duplicates
        report["exact_matches"] = self._count_exact_matches(real, synth)

        # 2. Compute DCR (Distance to Closest Record)
        dcr_values = self._compute_dcr(real, synth)

        if dcr_values is not None and len(dcr_values) > 0:
            report["dcr_stats"] = {
                "mean": float(np.mean(dcr_values)),
                "median": float(np.median(dcr_values)),
                "min": float(np.min(dcr_values)),
                "max": float(np.max(dcr_values)),
                "std": float(np.std(dcr_values)),
                "p5": float(np.percentile(dcr_values, 5)),
                "p25": float(np.percentile(dcr_values, 25)),
            }

            # Count records below threshold
            report["records_below_threshold"] = int(
                np.sum(dcr_values < self.dcr_threshold)
            )
            report["pct_below_threshold"] = float(
                report["records_below_threshold"] / len(dcr_values) * 100
            )

        # 3. Compute overall privacy score
        report["privacy_score"] = self._compute_privacy_score(report)

        logger.info(
            f"Privacy evaluation complete. Score: {report['privacy_score']:.4f}, "
            f"Exact matches: {report['exact_matches']}, "
            f"Below threshold: {report['records_below_threshold']}"
        )
        return report

    def _count_exact_matches(self, real: pd.DataFrame, synth: pd.DataFrame) -> int:
        """Count synthetic records that are exact copies of real records."""
        try:
            # Convert both to string for comparison (handles mixed types)
            real_strings = real.astype(str).apply(lambda row: "|".join(row), axis=1)
            synth_strings = synth.astype(str).apply(lambda row: "|".join(row), axis=1)
            matches = synth_strings.isin(real_strings).sum()
            if matches > 0:
                logger.warning(f"Found {matches} exact duplicate records!")
            return int(matches)
        except Exception as e:
            logger.error(f"Error checking exact matches: {e}")
            return -1

    def _compute_dcr(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """
        Compute Distance to Closest Record for each synthetic record.
        Uses Euclidean distance on normalized numeric features.
        """
        # Select and encode features
        numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = real.select_dtypes(include=["object", "category"]).columns.tolist()

        if not numeric_cols and not cat_cols:
            logger.warning("No columns available for DCR computation")
            return None

        # Encode categorical columns
        encoded_real = real[numeric_cols].copy()
        encoded_synth = synth[numeric_cols].copy()

        for col in cat_cols:
            le = LabelEncoder()
            # Fit on combined to handle categories in both datasets
            combined = pd.concat([real[col], synth[col]], ignore_index=True).astype(str)
            le.fit(combined)
            encoded_real[col] = le.transform(real[col].astype(str))
            encoded_synth[col] = le.transform(synth[col].astype(str))

        # Fill NaN and scale
        encoded_real = encoded_real.fillna(0)
        encoded_synth = encoded_synth.fillna(0)

        scaler = StandardScaler()
        scaler.fit(encoded_real)
        real_scaled = scaler.transform(encoded_real)
        synth_scaled = scaler.transform(encoded_synth)

        # Sample if datasets are too large (pairwise distance is O(n*m))
        if len(real_scaled) > self.sample_size:
            idx = np.random.choice(len(real_scaled), self.sample_size, replace=False)
            real_scaled = real_scaled[idx]

        if len(synth_scaled) > self.sample_size:
            idx = np.random.choice(len(synth_scaled), self.sample_size, replace=False)
            synth_scaled = synth_scaled[idx]

        logger.info(
            f"Computing DCR: {len(synth_scaled)} synthetic × {len(real_scaled)} real records"
        )

        # Compute pairwise distances
        distances = cdist(synth_scaled, real_scaled, metric="euclidean")

        # DCR = minimum distance for each synthetic record
        dcr_values = distances.min(axis=1)

        # Normalize by the average distance (so threshold is interpretable)
        avg_dist = distances.mean()
        if avg_dist > 0:
            dcr_values = dcr_values / avg_dist

        return dcr_values

    def _compute_privacy_score(self, report: dict) -> float:
        """
        Compute overall privacy score from 0 (bad) to 1 (good).

        Scoring:
        - Deduct heavily for exact matches
        - Score based on DCR distribution
        - Penalize for records below threshold
        """
        score = 1.0

        # Exact matches: heavy penalty
        exact = report.get("exact_matches", 0)
        if exact > 0:
            score -= min(0.5, exact * 0.01)  # Each match costs 1%, up to 50%

        # DCR stats
        dcr_stats = report.get("dcr_stats", {})
        if dcr_stats:
            # Reward high median DCR
            median_dcr = dcr_stats.get("median", 0)
            if median_dcr < self.dcr_threshold:
                score -= 0.3
            elif median_dcr < self.dcr_threshold * 2:
                score -= 0.1

            # Penalize records below threshold
            pct_below = report.get("pct_below_threshold", 0)
            score -= min(0.3, pct_below * 0.01)  # 1% per percent below threshold

        return max(0.0, min(1.0, score))


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python -m src.validation.privacy <real_csv> <synthetic_csv>")
        sys.exit(1)

    real_df = pd.read_csv(sys.argv[1])
    synthetic_df = pd.read_csv(sys.argv[2])

    evaluator = PrivacyEvaluator()
    report = evaluator.evaluate(real_df, synthetic_df)
    print(json.dumps(report, indent=2))
