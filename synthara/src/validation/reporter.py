"""
Synthara — Validation Reporter
Combines fidelity and privacy evaluation into comprehensive reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .fidelity import FidelityEvaluator
from .privacy import PrivacyEvaluator

logger = logging.getLogger(__name__)


class ValidationReporter:
    """
    Generates comprehensive validation reports combining fidelity
    and privacy metrics.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fidelity_evaluator = FidelityEvaluator()
        self.privacy_evaluator = PrivacyEvaluator()

    def generate_report(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        model_name: str = "unknown",
        metadata: Optional[object] = None,
        save: bool = True,
    ) -> dict:
        """
        Generate a full validation report.

        Args:
            real_df: Real/source DataFrame.
            synthetic_df: Synthetic DataFrame.
            model_name: Name of the model that generated the synthetic data.
            metadata: Optional SDV metadata object.
            save: Whether to save the report to disk.

        Returns:
            Complete validation report dictionary.
        """
        logger.info(f"Generating validation report for model '{model_name}'...")

        report = {
            "metadata": {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "real_shape": list(real_df.shape),
                "synthetic_shape": list(synthetic_df.shape),
            },
            "fidelity": self.fidelity_evaluator.evaluate(real_df, synthetic_df, metadata),
            "privacy": self.privacy_evaluator.evaluate(real_df, synthetic_df),
        }

        # Overall verdict
        fidelity_score = report["fidelity"]["overall_score"]
        privacy_score = report["privacy"]["privacy_score"]
        report["overall"] = {
            "fidelity_score": fidelity_score,
            "privacy_score": privacy_score,
            "combined_score": (fidelity_score + privacy_score) / 2,
            "verdict": self._verdict(fidelity_score, privacy_score),
        }

        if save:
            self._save_report(report, model_name)

        logger.info(
            f"Report complete: fidelity={fidelity_score:.4f}, "
            f"privacy={privacy_score:.4f}, "
            f"verdict={report['overall']['verdict']}"
        )
        return report

    def _verdict(self, fidelity: float, privacy: float) -> str:
        """Generate a human-readable verdict."""
        if fidelity >= 0.85 and privacy >= 0.85:
            return "EXCELLENT — High fidelity with strong privacy protection"
        elif fidelity >= 0.7 and privacy >= 0.7:
            return "GOOD — Acceptable for most use cases"
        elif fidelity >= 0.5 and privacy >= 0.5:
            return "FAIR — May need tuning for production use"
        elif privacy < 0.5:
            return "WARNING — Privacy concerns detected, review before use"
        else:
            return "POOR — Significant improvements needed"

    def _save_report(self, report: dict, model_name: str):
        """Save report as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{model_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {filepath}")
