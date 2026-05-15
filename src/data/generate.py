"""
Synthetic CNP fraud dataset generator for a UK BNPL / e-commerce payment processor.

Fraud injection rules reflect real UK CNP patterns:
- Electronics and travel merchants carry the highest CNP fraud rates
- Fraud transactions concentrate between 01:00–05:00 UTC
- Fraudulent cards show velocity spikes (multiple txns in short windows)
- Fraud amounts deviate significantly from the cardholder's historical average
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Merchant categories with (avg_amount_gbp, std_amount_gbp, cnp_fraud_base_rate)
MERCHANT_CATEGORIES = {
    "electronics":    (320.0,  180.0, 0.0045),
    "travel":         (510.0,  290.0, 0.0038),
    "fashion":        (85.0,   55.0,  0.0015),
    "grocery":        (42.0,   22.0,  0.0004),
    "gaming":         (35.0,   20.0,  0.0025),
    "jewellery":      (420.0,  250.0, 0.0040),
    "food_delivery":  (28.0,   12.0,  0.0003),
    "subscriptions":  (12.0,    5.0,  0.0008),
    "furniture":      (380.0,  200.0, 0.0018),
    "beauty":         (55.0,   30.0,  0.0010),
}

# Fraud hour weights: index = hour (0–23), higher = more fraud
_FRAUD_HOUR_WEIGHTS = np.array([
    0.8, 2.5, 4.0, 5.0, 4.5, 3.0,   # 00–05: peak fraud window
    1.5, 0.8, 0.5, 0.4, 0.4, 0.5,   # 06–11
    0.6, 0.6, 0.5, 0.5, 0.6, 0.7,   # 12–17
    0.8, 0.9, 1.0, 1.1, 1.0, 0.9,   # 18–23
])
_FRAUD_HOUR_WEIGHTS /= _FRAUD_HOUR_WEIGHTS.sum()

_LEGIT_HOUR_WEIGHTS = np.array([
    0.2, 0.1, 0.1, 0.1, 0.1, 0.2,
    0.8, 1.5, 2.5, 3.5, 4.0, 4.2,
    4.5, 4.3, 4.0, 3.8, 3.5, 3.0,
    2.8, 2.5, 2.0, 1.5, 0.8, 0.4,
])
_LEGIT_HOUR_WEIGHTS /= _LEGIT_HOUR_WEIGHTS.sum()


def generate_dataset(
    n_transactions: int = 600_000,
    n_cards: int = 50_000,
    n_days: int = 90,
    seed: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic CNP fraud dataset.

    Each row represents a single transaction. Columns include raw features
    used downstream by the feature engineering pipeline.

    Args:
        n_transactions: Total transactions to generate.
        n_cards: Number of unique card tokens.
        n_days: Simulation window in days (transactions span [0, n_days)).
        seed: Random seed for reproducibility.
        output_path: If provided, saves the CSV to this path.

    Returns:
        DataFrame with one transaction per row.
    """
    rng = np.random.default_rng(seed)
    categories = list(MERCHANT_CATEGORIES.keys())

    # --- Card pool: each card gets a base spending profile ---
    card_ids = [f"card_{i:06d}" for i in range(n_cards)]
    card_avg_amount = rng.lognormal(mean=4.2, sigma=0.7, size=n_cards)  # ~£67 median
    card_avg_amount = np.clip(card_avg_amount, 5.0, 2000.0)

    # --- Assign category and timestamp to each transaction ---
    cat_indices = rng.integers(0, len(categories), size=n_transactions)
    cat_names = [categories[i] for i in cat_indices]
    cat_fraud_rates = np.array([MERCHANT_CATEGORIES[c][2] for c in cat_names])
    cat_avg_amounts = np.array([MERCHANT_CATEGORIES[c][0] for c in cat_names])
    cat_std_amounts = np.array([MERCHANT_CATEGORIES[c][1] for c in cat_names])

    # Transaction timestamps (seconds offset from epoch start)
    timestamps_seconds = rng.uniform(0, n_days * 86400, size=n_transactions)
    timestamps_seconds.sort()
    hours = ((timestamps_seconds % 86400) / 3600).astype(int)

    # --- Determine fraud label ---
    # Fraud probability raised by: merchant risk + off-hour multiplier
    hour_fraud_multiplier = np.where(
        (hours >= 1) & (hours <= 5), 3.5, 1.0
    )
    fraud_prob = np.clip(cat_fraud_rates * hour_fraud_multiplier, 0, 0.15)
    is_fraud = rng.random(n_transactions) < fraud_prob

    # --- Assign card to each transaction ---
    # Fraudulent txns are concentrated on a small subset of compromised cards
    n_compromised = max(1, int(n_cards * 0.02))  # 2% of cards are compromised
    compromised_cards = rng.choice(n_cards, size=n_compromised, replace=False)

    card_indices = np.where(
        is_fraud,
        rng.choice(compromised_cards, size=n_transactions),
        rng.integers(0, n_cards, size=n_transactions),
    )

    # --- Transaction amounts ---
    # Fraud: amount is inflated (2–8× the card's historical average)
    fraud_multiplier = rng.uniform(2.0, 8.0, size=n_transactions)
    legit_noise = rng.normal(1.0, 0.25, size=n_transactions)
    legit_noise = np.clip(legit_noise, 0.3, 3.0)

    base_amounts = cat_avg_amounts + rng.normal(0, cat_std_amounts)
    amounts = np.where(
        is_fraud,
        card_avg_amount[card_indices] * fraud_multiplier,
        base_amounts * legit_noise,
    )
    amounts = np.clip(amounts, 0.50, 5000.0).round(2)

    # --- Merchant IDs (realistic: many merchants per category) ---
    n_merchants = 2000
    merchant_ids = [f"merch_{i:05d}" for i in range(n_merchants)]
    txn_merchants = rng.choice(n_merchants, size=n_transactions)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "transaction_id":  [f"txn_{i:08d}" for i in range(n_transactions)],
        "timestamp_sec":   timestamps_seconds,
        "card_id":         [card_ids[i] for i in card_indices],
        "merchant_id":     [merchant_ids[i] for i in txn_merchants],
        "merchant_category": cat_names,
        "amount_gbp":      amounts,
        "hour_of_day":     hours,
        "day_of_week":     ((timestamps_seconds // 86400) % 7).astype(int),
        "card_avg_amount_30d": card_avg_amount[card_indices].round(2),
        "is_fraud":        is_fraud.astype(int),
    })

    fraud_rate = df["is_fraud"].mean()
    print(f"Generated {len(df):,} transactions | fraud rate: {fraud_rate:.4%}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    out = Path(__file__).parents[2] / "data" / "raw" / "transactions.csv"
    generate_dataset(output_path=str(out))
