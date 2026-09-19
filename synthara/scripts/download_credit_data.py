"""
Synthara — Credit Scoring Dataset Generator
Since the Zindi credit scoring dataset requires authentication and competition terms agreement,
this script generates a highly realistic mock credit dataset reflecting African microfinance 
realities (informal income, mobile money history, group loans, etc.).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "source"


def create_credit_dataset(n_rows: int = 15000) -> str:
    """
    Generate a realistic credit scoring dataset.
    """
    logger.info(f"Generating African Microfinance Credit dataset ({n_rows} rows)...")
    np.random.seed(42)
    
    # 1. Demographics
    age = np.random.normal(loc=35, scale=10, size=n_rows).clip(18, 75).astype(int)
    gender = np.random.choice(["M", "F"], size=n_rows, p=[0.55, 0.45])
    location_type = np.random.choice(["Urban", "Peri-Urban", "Rural"], size=n_rows, p=[0.4, 0.3, 0.3])
    
    # 2. Employment & Income (Heavy informal sector representation)
    employment_type = np.random.choice(
        ["Formal", "Informal_Business", "Agriculture", "Casual_Labor"], 
        size=n_rows, p=[0.2, 0.5, 0.2, 0.1]
    )
    
    monthly_income = np.zeros(n_rows)
    for i, emp in enumerate(employment_type):
        if emp == "Formal":
            monthly_income[i] = np.random.lognormal(mean=11.5, sigma=0.8) # Higher, more stable
        elif emp == "Informal_Business":
            monthly_income[i] = np.random.lognormal(mean=11.0, sigma=1.2) # High variance
        elif emp == "Agriculture":
            monthly_income[i] = np.random.lognormal(mean=10.2, sigma=1.0) # Seasonal/lower
        else:
            monthly_income[i] = np.random.lognormal(mean=9.5, sigma=0.8)  # Lowest
    
    monthly_income = np.clip(monthly_income, 5000, 1_000_000).round(-2)
    
    # 3. Mobile Money Activity (Crucial alternative data)
    momo_active_months = np.random.randint(1, 60, size=n_rows)
    momo_monthly_tx_volume = (monthly_income * np.random.uniform(0.1, 1.5, size=n_rows)).round(-2)
    
    # 4. Loan Details
    loan_purpose = np.random.choice(
        ["Business_Inventory", "Agriculture_Inputs", "Education", "Medical", "Personal"],
        size=n_rows, p=[0.45, 0.20, 0.15, 0.10, 0.10]
    )
    
    # Loan amount correlated with income and purpose
    loan_amount_ratio = np.random.uniform(0.5, 3.0, size=n_rows)
    loan_amount = (monthly_income * loan_amount_ratio).round(-3)
    loan_amount = np.clip(loan_amount, 10_000, 2_000_000)
    
    loan_term_months = np.random.choice([3, 6, 9, 12, 18, 24], size=n_rows, p=[0.2, 0.3, 0.2, 0.2, 0.05, 0.05])
    interest_rate = np.random.uniform(1.5, 5.0, size=n_rows).round(1) # Monthly interest rate typical in MFI (1.5% - 5%)
    
    # 5. Calculate default probability based on features
    # Base log-odds
    log_odds = -2.0 
    
    # Risk factors
    log_odds += np.where(employment_type == "Casual_Labor", 1.0, 0)
    log_odds += np.where(employment_type == "Agriculture", 0.5, 0)
    log_odds -= np.where(employment_type == "Formal", 0.8, 0)
    
    log_odds += np.where(loan_amount / (monthly_income + 1) > 2.0, 1.2, 0) # Overleveraged
    log_odds -= np.where(momo_monthly_tx_volume / (loan_amount + 1) > 0.5, 0.7, 0) # High cashflow relative to loan
    log_odds -= (age - 35) * 0.02 # Older slightly less risky
    log_odds += np.where(location_type == "Rural", 0.3, 0) # Higher cost to serve/monitor
    
    # Add noise
    log_odds += np.random.normal(0, 1.5, size=n_rows)
    
    # Convert to probability and then to binary label
    prob_default = 1 / (1 + np.exp(-log_odds))
    is_default = (np.random.random(n_rows) < prob_default).astype(int)
    
    df = pd.DataFrame({
        "client_id": [f"CUST{i:06d}" for i in range(1, n_rows + 1)],
        "age": age,
        "gender": gender,
        "location_type": location_type,
        "employment_type": employment_type,
        "monthly_income": monthly_income,
        "momo_active_months": momo_active_months,
        "momo_monthly_tx_volume": momo_monthly_tx_volume,
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "default": is_default
    })
    
    output_path = DATA_DIR / "credit_scoring_sample.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    default_rate = df["default"].mean() * 100
    logger.info(f"Credit dataset generated at {output_path}")
    logger.info(f"Default rate: {default_rate:.2f}%")
    
    return str(output_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_credit_dataset()
