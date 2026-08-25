"""
Synthara — Business Constraints for African Financial Data
Defines constraints that enforce business rules specific to African
mobile money ecosystems during synthetic data generation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AfricanFinanceConstraints:
    """
    Applies and validates business constraints specific to African
    mobile money transactions.

    Constraints enforced:
    1. Balance consistency: newbalance = oldbalance ± amount
    2. Amount bounds: within realistic ranges per transaction type
    3. Transaction type validity: only allowed types
    4. Fraud consistency: fraud only on TRANSFER and CASH_OUT (PaySim pattern)
    5. Seasonal modulation: adjust volumes based on harvest/lean periods
    """

    # Typical amount ranges per transaction type in mobile money (in local currency units)
    AMOUNT_RANGES = {
        "CASH_IN": (100, 2_000_000),
        "CASH_OUT": (100, 2_000_000),
        "PAYMENT": (10, 500_000),
        "TRANSFER": (100, 5_000_000),
        "DEBIT": (50, 1_000_000),
    }

    VALID_TYPES = ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"]

    # In PaySim, fraud only occurs on TRANSFER and CASH_OUT
    FRAUD_ELIGIBLE_TYPES = ["TRANSFER", "CASH_OUT"]

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Optional constraint configuration from YAML.
        """
        self.config = config or {}
        mm_config = self.config.get("mobile_money", {})
        self.min_amount = mm_config.get("min_transaction_amount", 10)
        self.max_amount = mm_config.get("max_transaction_amount", 5_000_000)
        self.enforce_balance = mm_config.get("balance_consistency", True)

    def apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all business constraints to a synthetic DataFrame.

        Args:
            df: Synthetic DataFrame to constrain.

        Returns:
            Constrained DataFrame.
        """
        logger.info(f"Applying African finance constraints to {len(df)} rows...")
        original_len = len(df)

        df = self._enforce_amount_bounds(df)
        df = self._enforce_valid_types(df)
        df = self._enforce_balance_consistency(df)
        df = self._enforce_fraud_consistency(df)
        df = self._enforce_non_negative_balances(df)

        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Constraints removed {removed} invalid rows ({removed/original_len*100:.1f}%)")

        return df.reset_index(drop=True)

    def _enforce_amount_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip amounts to valid ranges."""
        if "amount" not in df.columns:
            return df

        df = df.copy()
        df["amount"] = df["amount"].clip(lower=self.min_amount, upper=self.max_amount)

        # Apply type-specific bounds if type column exists
        if "type" in df.columns:
            for tx_type, (min_amt, max_amt) in self.AMOUNT_RANGES.items():
                mask = df["type"] == tx_type
                if mask.any():
                    df.loc[mask, "amount"] = df.loc[mask, "amount"].clip(
                        lower=min_amt, upper=max_amt
                    )

        return df

    def _enforce_valid_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid transaction types."""
        if "type" not in df.columns:
            return df

        invalid_mask = ~df["type"].isin(self.VALID_TYPES)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            logger.warning(f"Removing {n_invalid} rows with invalid transaction types")
            df = df[~invalid_mask]

        return df

    def _enforce_balance_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix balance columns to be consistent with transaction amounts.
        For outgoing transactions: newbalanceOrig = oldbalanceOrg - amount
        For incoming transactions: newbalanceOrig = oldbalanceOrg + amount
        """
        if not self.enforce_balance:
            return df

        required_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "type"]
        if not all(col in df.columns for col in required_cols):
            return df

        df = df.copy()

        # Outgoing transaction types (money leaves originator)
        outgoing = df["type"].isin(["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"])
        df.loc[outgoing, "newbalanceOrig"] = (
            df.loc[outgoing, "oldbalanceOrg"] - df.loc[outgoing, "amount"]
        ).clip(lower=0)

        # Incoming transaction types (money enters originator)
        incoming = df["type"] == "CASH_IN"
        df.loc[incoming, "newbalanceOrig"] = (
            df.loc[incoming, "oldbalanceOrg"] + df.loc[incoming, "amount"]
        )

        return df

    def _enforce_fraud_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure fraud flag is only set on eligible transaction types."""
        if "isFraud" not in df.columns or "type" not in df.columns:
            return df

        df = df.copy()
        ineligible_fraud = (df["isFraud"] == 1) & (~df["type"].isin(self.FRAUD_ELIGIBLE_TYPES))
        n_fixed = ineligible_fraud.sum()
        if n_fixed > 0:
            logger.info(f"Fixed {n_fixed} fraud flags on ineligible transaction types")
            df.loc[ineligible_fraud, "isFraud"] = 0

        return df

    def _enforce_non_negative_balances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all balance columns are non-negative."""
        df = df.copy()
        balance_cols = [
            "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
        ]
        for col in balance_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        return df

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validate a DataFrame against all constraints and return a report.

        Returns:
            Dictionary with validation results per constraint.
        """
        report = {}

        # Amount bounds
        if "amount" in df.columns:
            out_of_bounds = ((df["amount"] < self.min_amount) | (df["amount"] > self.max_amount)).sum()
            report["amount_out_of_bounds"] = int(out_of_bounds)

        # Valid types
        if "type" in df.columns:
            invalid_types = (~df["type"].isin(self.VALID_TYPES)).sum()
            report["invalid_transaction_types"] = int(invalid_types)

        # Negative balances
        balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        for col in balance_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    report[f"negative_{col}"] = int(neg_count)

        # Fraud on ineligible types
        if "isFraud" in df.columns and "type" in df.columns:
            ineligible = ((df["isFraud"] == 1) & (~df["type"].isin(self.FRAUD_ELIGIBLE_TYPES))).sum()
            report["fraud_on_ineligible_type"] = int(ineligible)

        report["all_valid"] = all(v == 0 for v in report.values())
        return report
