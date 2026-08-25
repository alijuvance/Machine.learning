"""
Synthara — African Financial Patterns Detector
Identifies and quantifies patterns specific to African mobile money ecosystems:
- Transaction type distributions (CASH-IN/CASH-OUT ratios)
- Temporal cycles (hourly, daily, monthly patterns)
- Amount distributions by transaction type
- Seasonal patterns (harvest vs lean periods)
- Fraud patterns specific to mobile money
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MobileMoneyPatterns:
    """Container for detected mobile money patterns."""

    # Transaction type distribution
    transaction_type_distribution: dict = field(default_factory=dict)

    # Temporal patterns
    hourly_distribution: Optional[pd.Series] = None
    daily_distribution: Optional[pd.Series] = None
    monthly_volume_trend: Optional[pd.Series] = None

    # Amount patterns by type
    amount_stats_by_type: Optional[pd.DataFrame] = None

    # Balance patterns
    avg_balance_by_type: dict = field(default_factory=dict)
    balance_consistency_violations: int = 0

    # Fraud patterns
    fraud_rate: float = 0.0
    fraud_by_type: dict = field(default_factory=dict)
    fraud_amount_stats: dict = field(default_factory=dict)
    fraud_temporal_pattern: Optional[pd.Series] = None

    # Seasonality indicators
    seasonality_detected: bool = False
    peak_periods: list = field(default_factory=list)
    lean_periods: list = field(default_factory=list)
    seasonality_strength: float = 0.0

    def summary(self) -> dict:
        """Return a JSON-serializable summary of detected patterns."""
        return {
            "transaction_types": self.transaction_type_distribution,
            "fraud_rate": round(self.fraud_rate * 100, 4),
            "fraud_rate_pct": f"{self.fraud_rate * 100:.4f}%",
            "fraud_by_type": self.fraud_by_type,
            "balance_violations": self.balance_consistency_violations,
            "seasonality_detected": self.seasonality_detected,
            "seasonality_strength": round(self.seasonality_strength, 4),
            "peak_periods": self.peak_periods,
            "lean_periods": self.lean_periods,
        }


class AfricanPatternsDetector:
    """
    Detects and quantifies financial patterns specific to African
    mobile money ecosystems.

    Designed for PaySim-like datasets but adaptable to any mobile money
    transaction dataset with columns like: step, type, amount, balances, isFraud.
    """

    # PaySim column mapping (can be overridden)
    DEFAULT_COLUMNS = {
        "step": "step",            # Time step (1 step = 1 hour in PaySim)
        "type": "type",            # Transaction type
        "amount": "amount",        # Transaction amount
        "old_balance_orig": "oldbalanceOrg",
        "new_balance_orig": "newbalanceOrig",
        "old_balance_dest": "oldbalanceDest",
        "new_balance_dest": "newbalanceDest",
        "is_fraud": "isFraud",
        "name_orig": "nameOrig",
        "name_dest": "nameDest",
    }

    def __init__(self, column_mapping: Optional[dict] = None):
        """
        Args:
            column_mapping: Override default column name mapping.
        """
        self.columns = {**self.DEFAULT_COLUMNS}
        if column_mapping:
            self.columns.update(column_mapping)

    def detect_patterns(self, df: pd.DataFrame) -> MobileMoneyPatterns:
        """
        Run full pattern detection pipeline.

        Args:
            df: DataFrame with mobile money transaction data.

        Returns:
            MobileMoneyPatterns with all detected patterns.
        """
        logger.info(f"Detecting African financial patterns on {len(df)} transactions...")
        patterns = MobileMoneyPatterns()

        # Transaction type distribution
        type_col = self.columns["type"]
        if type_col in df.columns:
            patterns.transaction_type_distribution = (
                df[type_col].value_counts(normalize=True).to_dict()
            )
            logger.info(f"Transaction types: {patterns.transaction_type_distribution}")

        # Amount patterns by type
        patterns.amount_stats_by_type = self._amount_by_type(df)

        # Temporal patterns
        self._detect_temporal_patterns(df, patterns)

        # Balance consistency
        patterns.balance_consistency_violations = self._check_balance_consistency(df)

        # Fraud patterns
        self._detect_fraud_patterns(df, patterns)

        # Seasonality
        self._detect_seasonality(df, patterns)

        logger.info(f"Pattern detection complete: {patterns.summary()}")
        return patterns

    def _amount_by_type(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute amount statistics grouped by transaction type."""
        type_col = self.columns["type"]
        amount_col = self.columns["amount"]

        if type_col not in df.columns or amount_col not in df.columns:
            return None

        return df.groupby(type_col)[amount_col].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )

    def _detect_temporal_patterns(self, df: pd.DataFrame, patterns: MobileMoneyPatterns):
        """Detect hourly, daily, and monthly transaction patterns."""
        step_col = self.columns["step"]
        if step_col not in df.columns:
            return

        # In PaySim, 1 step = 1 hour, total 744 steps = 31 days
        # Convert steps to hour-of-day (0-23)
        df_temp = df.copy()
        df_temp["hour_of_day"] = df_temp[step_col] % 24
        df_temp["day"] = df_temp[step_col] // 24

        patterns.hourly_distribution = df_temp["hour_of_day"].value_counts().sort_index()
        patterns.daily_distribution = df_temp["day"].value_counts().sort_index()

        # Monthly aggregation (approximate: 30 days)
        df_temp["week"] = df_temp["day"] // 7
        patterns.monthly_volume_trend = df_temp.groupby("week").size()

    def _check_balance_consistency(self, df: pd.DataFrame) -> int:
        """
        Check balance consistency: for non-merchant accounts,
        newbalance should roughly equal oldbalance +/- amount.
        Returns count of violations.
        """
        amount_col = self.columns["amount"]
        old_bal = self.columns["old_balance_orig"]
        new_bal = self.columns["new_balance_orig"]

        if not all(col in df.columns for col in [amount_col, old_bal, new_bal]):
            return 0

        # Only check rows where balances are non-zero (merchants have 0 balances)
        mask = (df[old_bal] > 0) | (df[new_bal] > 0)
        subset = df[mask]

        if len(subset) == 0:
            return 0

        # Expected new balance for outgoing transactions
        expected_debit = subset[old_bal] - subset[amount_col]
        expected_credit = subset[old_bal] + subset[amount_col]

        # Check if actual new balance matches either expected
        tolerance = 1.0  # Allow small rounding differences
        is_consistent = (
            (np.abs(subset[new_bal] - expected_debit) < tolerance)
            | (np.abs(subset[new_bal] - expected_credit) < tolerance)
            | (subset[new_bal] == 0)  # Zeroed out accounts
        )

        violations = int((~is_consistent).sum())
        if violations > 0:
            logger.warning(
                f"Found {violations} balance consistency violations "
                f"({violations / len(subset) * 100:.2f}% of non-merchant transactions)"
            )
        return violations

    def _detect_fraud_patterns(self, df: pd.DataFrame, patterns: MobileMoneyPatterns):
        """Analyze fraud-specific patterns in the dataset."""
        fraud_col = self.columns["is_fraud"]
        type_col = self.columns["type"]
        amount_col = self.columns["amount"]
        step_col = self.columns["step"]

        if fraud_col not in df.columns:
            return

        # Overall fraud rate
        patterns.fraud_rate = float(df[fraud_col].mean())

        # Fraud rate by transaction type
        if type_col in df.columns:
            fraud_by_type = df.groupby(type_col)[fraud_col].mean()
            patterns.fraud_by_type = {
                k: round(float(v) * 100, 4) for k, v in fraud_by_type.items()
            }

        # Fraud amount statistics
        if amount_col in df.columns:
            fraud_amounts = df[df[fraud_col] == 1][amount_col]
            legit_amounts = df[df[fraud_col] == 0][amount_col]
            patterns.fraud_amount_stats = {
                "fraud_mean": float(fraud_amounts.mean()) if len(fraud_amounts) > 0 else 0,
                "fraud_median": float(fraud_amounts.median()) if len(fraud_amounts) > 0 else 0,
                "legit_mean": float(legit_amounts.mean()) if len(legit_amounts) > 0 else 0,
                "legit_median": float(legit_amounts.median()) if len(legit_amounts) > 0 else 0,
                "fraud_count": int(fraud_amounts.shape[0]),
                "legit_count": int(legit_amounts.shape[0]),
            }

        # Fraud temporal pattern
        if step_col in df.columns:
            fraud_df = df[df[fraud_col] == 1]
            fraud_df_temp = fraud_df.copy()
            fraud_df_temp["hour_of_day"] = fraud_df_temp[step_col] % 24
            patterns.fraud_temporal_pattern = (
                fraud_df_temp["hour_of_day"].value_counts().sort_index()
            )

    def _detect_seasonality(self, df: pd.DataFrame, patterns: MobileMoneyPatterns):
        """
        Detect seasonal patterns in transaction volumes.
        Uses coefficient of variation of weekly volumes as a proxy.
        """
        step_col = self.columns["step"]
        if step_col not in df.columns:
            return

        df_temp = df.copy()
        df_temp["day"] = df_temp[step_col] // 24
        df_temp["week"] = df_temp["day"] // 7

        weekly_volumes = df_temp.groupby("week").size()

        if len(weekly_volumes) < 3:
            return

        # Coefficient of variation as seasonality strength
        cv = float(weekly_volumes.std() / weekly_volumes.mean()) if weekly_volumes.mean() > 0 else 0.0
        patterns.seasonality_strength = cv

        # Detect if seasonality is significant (CV > 0.1 = 10% variation)
        if cv > 0.1:
            patterns.seasonality_detected = True
            median_volume = weekly_volumes.median()
            patterns.peak_periods = [
                int(w) for w in weekly_volumes[weekly_volumes > median_volume * 1.2].index
            ]
            patterns.lean_periods = [
                int(w) for w in weekly_volumes[weekly_volumes < median_volume * 0.8].index
            ]
        else:
            patterns.seasonality_detected = False

        logger.info(
            f"Seasonality: detected={patterns.seasonality_detected}, "
            f"strength={cv:.4f}, peaks={patterns.peak_periods}, "
            f"leans={patterns.lean_periods}"
        )


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.data_analysis.african_patterns <csv_path> [--nrows N]")
        sys.exit(1)

    csv_path = sys.argv[1]
    nrows = None
    if "--nrows" in sys.argv:
        nrows = int(sys.argv[sys.argv.index("--nrows") + 1])

    df = pd.read_csv(csv_path, nrows=nrows)
    detector = AfricanPatternsDetector()
    patterns = detector.detect_patterns(df)
    print(json.dumps(patterns.summary(), indent=2))
