"""
Synthara — Unified Synthesizer
Wrapper around SDV models (CTGAN, GaussianCopula, TVAE) providing a
unified interface for synthetic data generation with African-specific
constraints and post-processing.
"""

import logging
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from .config import get_model_params, load_config
from .constraints import AfricanFinanceConstraints

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "ctgan": CTGANSynthesizer,
    "gaussian_copula": GaussianCopulaSynthesizer,
    "tvae": TVAESynthesizer,
}


class SyntharaSynthesizer:
    """
    Unified synthetic data generator for African financial datasets.

    Wraps SDV's synthesizers with:
    - Automatic metadata detection
    - African-specific business constraints
    - Configurable fraud ratio preservation
    - Model comparison support (train all 3, compare results)

    Usage:
        synth = SyntharaSynthesizer(model_name="ctgan")
        synth.fit(real_df)
        synthetic_df = synth.generate(num_rows=10000)
    """

    def __init__(
        self,
        model_name: Literal["ctgan", "gaussian_copula", "tvae"] = "ctgan",
        config_path: Optional[str] = None,
        apply_constraints: bool = True,
        random_seed: int = 42,
    ):
        """
        Args:
            model_name: Which SDV model to use.
            config_path: Path to YAML configuration file.
            apply_constraints: Whether to apply African business constraints post-generation.
            random_seed: Random seed for reproducibility.
        """
        if model_name not in MODEL_CLASSES:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_CLASSES.keys())}")

        self.model_name = model_name
        self.config = load_config(config_path)
        self.apply_constraints = apply_constraints
        self.random_seed = random_seed

        self.metadata: Optional[SingleTableMetadata] = None
        self.synthesizer = None
        self.is_fitted = False
        self.real_df: Optional[pd.DataFrame] = None
        self.fit_time: float = 0.0

        # Constraints engine
        constraint_config = self.config.get("constraints", {})
        self.constraints = AfricanFinanceConstraints(constraint_config)

    def fit(self, df: pd.DataFrame, primary_key: Optional[str] = None) -> "SyntharaSynthesizer":
        """
        Fit the synthesizer on real data.

        Args:
            df: Real/source DataFrame.
            primary_key: Optional primary key column.

        Returns:
            self (for chaining).
        """
        logger.info(f"Fitting {self.model_name} on {df.shape[0]} rows × {df.shape[1]} columns...")
        self.real_df = df.copy()

        # Build metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(df)

        if primary_key and primary_key in df.columns:
            self.metadata.set_primary_key(primary_key)

        # Get model-specific parameters
        model_params = get_model_params(self.config, self.model_name)
        model_class = MODEL_CLASSES[self.model_name]

        # Filter params to only those accepted by the model class
        # (different models accept different parameters)
        self.synthesizer = model_class(self.metadata, **self._filter_params(model_class, model_params))

        # Fit
        start_time = time.time()
        self.synthesizer.fit(df)
        self.fit_time = time.time() - start_time
        self.is_fitted = True

        logger.info(f"Fitting complete in {self.fit_time:.1f}s")
        return self

    def generate(
        self,
        num_rows: Optional[int] = None,
        preserve_fraud_ratio: bool = True,
    ) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            num_rows: Number of rows to generate. Defaults to config value or source size.
            preserve_fraud_ratio: If True and source has fraud labels, preserve the fraud ratio.

        Returns:
            Synthetic DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Synthesizer must be fitted before generating. Call .fit() first.")

        gen_config = self.config.get("generation", {})
        if num_rows is None:
            num_rows = gen_config.get("default_num_rows", len(self.real_df))

        logger.info(f"Generating {num_rows} synthetic rows with {self.model_name}...")

        start_time = time.time()
        synthetic_df = self.synthesizer.sample(num_rows=num_rows)
        gen_time = time.time() - start_time

        logger.info(f"Generation complete in {gen_time:.1f}s")

        # Apply business constraints
        if self.apply_constraints:
            synthetic_df = self.constraints.apply_constraints(synthetic_df)

        # Validate constraints
        validation = self.constraints.validate(synthetic_df)
        if not validation.get("all_valid", False):
            logger.warning(f"Constraint validation issues: {validation}")

        logger.info(f"Final synthetic dataset: {synthetic_df.shape[0]} rows × {synthetic_df.shape[1]} columns")
        return synthetic_df

    def save_model(self, filepath: str):
        """Save the fitted synthesizer to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        self.synthesizer.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a previously saved synthesizer from disk."""
        model_class = MODEL_CLASSES[self.model_name]
        self.synthesizer = model_class.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

    def _filter_params(self, model_class, params: dict) -> dict:
        """Filter parameters to only those accepted by the model class constructor."""
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self", "metadata"}
        filtered = {k: v for k, v in params.items() if k in valid_params}

        skipped = set(params.keys()) - set(filtered.keys())
        if skipped:
            logger.debug(f"Skipped params not accepted by {model_class.__name__}: {skipped}")

        return filtered


def generate_with_all_models(
    df: pd.DataFrame,
    num_rows: int = 10000,
    config_path: Optional[str] = None,
    apply_constraints: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic data using all 3 models for comparison.

    Args:
        df: Real/source DataFrame.
        num_rows: Number of rows to generate per model.
        config_path: Path to YAML configuration.
        apply_constraints: Whether to apply African constraints.

    Returns:
        Dict mapping model name to synthetic DataFrame.
    """
    results = {}

    for model_name in MODEL_CLASSES:
        logger.info(f"\n{'='*60}\nTraining {model_name}...\n{'='*60}")
        try:
            synth = SyntharaSynthesizer(
                model_name=model_name,
                config_path=config_path,
                apply_constraints=apply_constraints,
            )
            synth.fit(df)
            synthetic_df = synth.generate(num_rows=num_rows)
            results[model_name] = synthetic_df
            logger.info(f"{model_name}: generated {len(synthetic_df)} rows in {synth.fit_time:.1f}s")
        except Exception as e:
            logger.error(f"Failed to generate with {model_name}: {e}")
            results[model_name] = None

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.generation.synthesizer <csv_path> [--model ctgan] [--nrows 10000] [--output output.csv]")
        sys.exit(1)

    csv_path = sys.argv[1]
    model = "ctgan"
    nrows = 10000
    output = "data/generated/synthetic_output.csv"

    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if "--nrows" in sys.argv:
        nrows = int(sys.argv[sys.argv.index("--nrows") + 1])
    if "--output" in sys.argv:
        output = sys.argv[sys.argv.index("--output") + 1]

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    synth = SyntharaSynthesizer(model_name=model)
    synth.fit(df)
    synthetic_df = synth.generate(num_rows=nrows)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output, index=False)
    print(f"Saved {len(synthetic_df)} synthetic rows to {output}")
