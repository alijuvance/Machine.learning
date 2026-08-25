"""
Synthara — TSTR (Train Synthetic, Test Real) Pipeline
⭐ THE KEY DELIVERABLE: demonstrates that models trained on Synthara's
synthetic data perform comparably to models trained on real data.

Pipeline:
1. Split real data into train/test
2. Scenario A: Train on real, test on real (baseline)
3. Scenario B: Train on synthetic, test on real (the test)
4. Scenario C: Train on real+synthetic, test on real (augmentation)
5. Compare all scenarios across multiple ML models
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluator import evaluate_model, format_comparison_table
from .models import MODEL_REGISTRY, get_model, prepare_features

logger = logging.getLogger(__name__)


class TSTRBenchmark:
    """
    Train Synthetic, Test Real benchmark pipeline.

    This is the central proof-of-value for Synthara:
    if a model trained on synthetic data performs well on real data,
    the synthetic data captures the true underlying patterns.
    """

    def __init__(
        self,
        target_column: str = "isFraud",
        drop_columns: Optional[list] = None,
        encode_columns: Optional[list] = None,
        test_size: float = 0.2,
        random_seed: int = 42,
        model_names: Optional[list] = None,
    ):
        """
        Args:
            target_column: Name of the binary target column.
            drop_columns: Columns to exclude from features.
            encode_columns: Categorical columns to label-encode.
            test_size: Fraction of real data held out for testing.
            random_seed: Random seed for reproducibility.
            model_names: List of model names to benchmark.
        """
        self.target_column = target_column
        self.drop_columns = drop_columns or ["nameOrig", "nameDest", "isFlaggedFraud"]
        self.encode_columns = encode_columns or ["type"]
        self.test_size = test_size
        self.random_seed = random_seed
        self.model_names = model_names or list(MODEL_REGISTRY.keys())

    def run(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        augmented_ratio: float = 1.0,
    ) -> dict:
        """
        Run the complete TSTR benchmark.

        Args:
            real_df: Real/source DataFrame.
            synthetic_df: Synthetic DataFrame.
            augmented_ratio: Ratio of synthetic to real rows for augmented scenario.

        Returns:
            Nested dict: {scenario: {model: metrics}}.
        """
        logger.info("=" * 70)
        logger.info("STARTING TSTR BENCHMARK")
        logger.info("=" * 70)
        start_time = time.time()

        # 1. Split real data
        logger.info(f"Splitting real data: {len(real_df)} rows, test_size={self.test_size}")
        real_train, real_test = train_test_split(
            real_df,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=real_df[self.target_column] if self.target_column in real_df.columns else None,
        )
        logger.info(f"Real train: {len(real_train)}, Real test: {len(real_test)}")

        # Prepare test features (always real)
        X_test, y_test = prepare_features(
            real_test, self.target_column, self.drop_columns, self.encode_columns
        )

        results = {}

        # 2. Scenario A: Train on Real, Test on Real (baseline)
        logger.info("\n" + "=" * 50)
        logger.info("SCENARIO A: Train on Real → Test on Real")
        logger.info("=" * 50)
        X_train_real, y_train_real = prepare_features(
            real_train, self.target_column, self.drop_columns, self.encode_columns
        )
        # Align columns
        X_train_real, X_test_aligned = self._align_columns(X_train_real, X_test)
        results["Real only"] = self._evaluate_scenario(X_train_real, y_train_real, X_test_aligned, y_test)

        # 3. Scenario B: Train on Synthetic, Test on Real
        logger.info("\n" + "=" * 50)
        logger.info("SCENARIO B: Train on Synthetic → Test on Real")
        logger.info("=" * 50)
        X_train_synth, y_train_synth = prepare_features(
            synthetic_df, self.target_column, self.drop_columns, self.encode_columns
        )
        X_train_synth, X_test_aligned = self._align_columns(X_train_synth, X_test)
        results["Synthetic only"] = self._evaluate_scenario(X_train_synth, y_train_synth, X_test_aligned, y_test)

        # 4. Scenario C: Train on Real + Synthetic (augmented)
        logger.info("\n" + "=" * 50)
        logger.info("SCENARIO C: Train on Real + Synthetic → Test on Real")
        logger.info("=" * 50)
        n_synth_aug = int(len(real_train) * augmented_ratio)
        synth_sample = synthetic_df.sample(
            n=min(n_synth_aug, len(synthetic_df)),
            random_state=self.random_seed,
        )
        augmented_df = pd.concat([real_train, synth_sample], ignore_index=True)
        X_train_aug, y_train_aug = prepare_features(
            augmented_df, self.target_column, self.drop_columns, self.encode_columns
        )
        X_train_aug, X_test_aligned = self._align_columns(X_train_aug, X_test)
        results["Augmented (Real+Synth)"] = self._evaluate_scenario(X_train_aug, y_train_aug, X_test_aligned, y_test)

        # Print comparison table
        total_time = time.time() - start_time
        table = format_comparison_table(results)
        logger.info(f"\n{table}")
        logger.info(f"\nTotal benchmark time: {total_time:.1f}s")

        return results

    def _evaluate_scenario(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """Train and evaluate all models for a single scenario."""
        scenario_results = {}

        for model_name in self.model_names:
            logger.info(f"  Training {model_name}...")
            try:
                model = get_model(model_name)

                # Handle auto scale_pos_weight for XGBoost
                if hasattr(model, "scale_pos_weight") and model_name == "xgboost":
                    n_neg = (y_train == 0).sum()
                    n_pos = (y_train == 1).sum()
                    if n_pos > 0:
                        model.set_params(scale_pos_weight=n_neg / n_pos)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]

                metrics = evaluate_model(y_test, y_pred, y_proba, model_name)
                scenario_results[model_name] = metrics

                logger.info(
                    f"  {model_name}: AUC={metrics['roc_auc']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"Recall={metrics['recall']:.4f}"
                )

            except Exception as e:
                logger.error(f"  Failed to train {model_name}: {e}")
                scenario_results[model_name] = {"error": str(e)}

        return scenario_results

    def _align_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure train and test DataFrames have the same columns.
        """
        common_cols = sorted(set(train_df.columns) & set(test_df.columns))
        if not common_cols:
            raise ValueError("No common columns between train and test sets")

        missing_train = set(test_df.columns) - set(train_df.columns)
        missing_test = set(train_df.columns) - set(test_df.columns)

        if missing_train or missing_test:
            logger.info(
                f"Aligning columns: {len(common_cols)} common, "
                f"{len(missing_train)} missing in train, "
                f"{len(missing_test)} missing in test"
            )

        return train_df[common_cols], test_df[common_cols]


def run_benchmark(
    real_path: str,
    synthetic_path: str,
    target_column: str = "isFraud",
    nrows: Optional[int] = None,
) -> dict:
    """
    Convenience function to run the full TSTR benchmark from file paths.

    Args:
        real_path: Path to real data CSV.
        synthetic_path: Path to synthetic data CSV.
        target_column: Name of the target column.
        nrows: Optional max rows to load.

    Returns:
        Benchmark results dictionary.
    """
    real_df = pd.read_csv(real_path, nrows=nrows)
    synthetic_df = pd.read_csv(synthetic_path, nrows=nrows)

    benchmark = TSTRBenchmark(target_column=target_column)
    return benchmark.run(real_df, synthetic_df)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python -m src.benchmark.tstr <real_csv> <synthetic_csv> [--target isFraud] [--nrows N]")
        sys.exit(1)

    real_path = sys.argv[1]
    synth_path = sys.argv[2]
    target = "isFraud"
    nrows = None

    if "--target" in sys.argv:
        target = sys.argv[sys.argv.index("--target") + 1]
    if "--nrows" in sys.argv:
        nrows = int(sys.argv[sys.argv.index("--nrows") + 1])

    results = run_benchmark(real_path, synth_path, target, nrows)
    print(json.dumps(results, indent=2, default=str))
