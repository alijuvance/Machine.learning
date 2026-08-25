"""
Synthara — PaySim Dataset Downloader
Downloads the PaySim mobile money fraud detection dataset.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "source"


def download_paysim():
    """
    Download PaySim dataset using kagglehub.
    Requires Kaggle API credentials (kaggle.json).
    
    Alternative: manual download from https://www.kaggle.com/datasets/ealaxi/paysim1
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "paysim.csv"

    if output_path.exists():
        logger.info(f"PaySim dataset already exists at {output_path}")
        return str(output_path)

    try:
        import kagglehub
        logger.info("Downloading PaySim dataset from Kaggle...")
        path = kagglehub.dataset_download("ealaxi/paysim1")
        logger.info(f"Downloaded to: {path}")

        # kagglehub downloads to a cache directory, we need to find the CSV
        import shutil
        downloaded_dir = Path(path)

        # Look for the CSV file
        csv_files = list(downloaded_dir.rglob("*.csv"))
        if csv_files:
            src_file = csv_files[0]
            shutil.copy2(src_file, output_path)
            logger.info(f"Copied to {output_path}")
            return str(output_path)
        else:
            logger.error(f"No CSV files found in {downloaded_dir}")
            return None

    except ImportError:
        logger.warning(
            "kagglehub not installed. Install with: pip install kagglehub\n"
            "Or download manually from: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            f"Place the CSV file at: {output_path}"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to download PaySim: {e}")
        logger.info(
            f"\nManual download instructions:\n"
            f"1. Go to https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            f"2. Download the CSV file\n"
            f"3. Place it at: {output_path}\n"
        )
        return None


def create_sample_dataset(n_rows: int = 10000) -> str:
    """
    Create a small sample dataset for testing when PaySim is not available.
    Mimics PaySim's structure and basic patterns.
    """
    import numpy as np
    import pandas as pd

    logger.info(f"Creating sample mobile money dataset with {n_rows} rows...")
    np.random.seed(42)

    # Transaction types with realistic distribution
    types = np.random.choice(
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_rows,
        p=[0.35, 0.34, 0.22, 0.08, 0.01],
    )

    # Amounts based on transaction type
    amounts = np.zeros(n_rows)
    for i, t in enumerate(types):
        if t == "PAYMENT":
            amounts[i] = np.random.lognormal(mean=8, sigma=1.5)
        elif t in ["CASH_OUT", "CASH_IN"]:
            amounts[i] = np.random.lognormal(mean=10, sigma=2)
        elif t == "TRANSFER":
            amounts[i] = np.random.lognormal(mean=11, sigma=2.5)
        else:  # DEBIT
            amounts[i] = np.random.lognormal(mean=9, sigma=1)

    amounts = np.clip(amounts, 100, 5_000_000)

    # Balances
    old_balance_org = np.random.lognormal(mean=11, sigma=2, size=n_rows).clip(0, 10_000_000)
    new_balance_orig = np.maximum(0, old_balance_org - amounts * (types != "CASH_IN").astype(float)
                                  + amounts * (types == "CASH_IN").astype(float))

    old_balance_dest = np.random.lognormal(mean=11, sigma=2, size=n_rows).clip(0, 10_000_000)
    new_balance_dest = old_balance_dest + amounts

    # Fraud (only on TRANSFER and CASH_OUT, ~0.13% rate)
    is_fraud = np.zeros(n_rows, dtype=int)
    eligible = (types == "TRANSFER") | (types == "CASH_OUT")
    n_fraud = max(1, int(eligible.sum() * 0.003))
    fraud_indices = np.random.choice(np.where(eligible)[0], size=n_fraud, replace=False)
    is_fraud[fraud_indices] = 1

    # Fraud transactions tend to be larger
    amounts[fraud_indices] *= np.random.uniform(2, 10, size=n_fraud)

    df = pd.DataFrame({
        "step": np.random.randint(1, 744, size=n_rows),
        "type": types,
        "amount": amounts.round(2),
        "nameOrig": [f"C{i:010d}" for i in range(n_rows)],
        "oldbalanceOrg": old_balance_org.round(2),
        "newbalanceOrig": new_balance_orig.round(2),
        "nameDest": [f"C{i+n_rows:010d}" if t != "PAYMENT" else f"M{i:010d}" for i, t in enumerate(types)],
        "oldbalanceDest": old_balance_dest.round(2),
        "newbalanceDest": new_balance_dest.round(2),
        "isFraud": is_fraud,
        "isFlaggedFraud": 0,
    })

    # Sort by step (time)
    df = df.sort_values("step").reset_index(drop=True)

    output_path = DATA_DIR / "paysim_sample.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample dataset saved to {output_path} ({len(df)} rows, {is_fraud.sum()} fraud)")
    return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    result = download_paysim()
    if result is None:
        logger.info("\nFalling back to sample dataset generation...")
        result = create_sample_dataset()
        logger.info(f"\nSample dataset ready at: {result}")
    else:
        logger.info(f"\nDataset ready at: {result}")
