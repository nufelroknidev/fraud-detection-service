"""
Velocity and behavioural feature engineering for CNP fraud detection.

These features capture the rapid-fire card activity that distinguishes
a fraudster burning through a compromised card from normal spending:
  - Rolling counts/sums over 1 h, 6 h, 24 h, and 7 d per card
  - Merchant-level velocity over 1 h
  - Time elapsed since the card's previous transaction
  - Amount deviation from the card's 30-day baseline
  - Cyclic hour/day-of-week encodings

All rolling features use closed='left' so each row only sees transactions
that occurred strictly before it — zero future leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_BASE_TS = pd.Timestamp("2020-01-01")

_PROJECT_ROOT = Path(__file__).parents[2]
_RAW_PATH  = _PROJECT_ROOT / "data" / "raw"  / "transactions.csv"
_OUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "transactions_featured.csv"


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp and replace the index with a monotonic DatetimeIndex."""
    df = df.sort_values("timestamp_sec").copy()
    df.index = _BASE_TS + pd.to_timedelta(df["timestamp_sec"], unit="s")
    df.index.name = None
    return df


def _card_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card rolling transaction count and amount sum for 1 h, 6 h, 24 h, and 7 d."""
    grp = df.groupby("card_id", sort=False)
    for window in ("1h", "6h", "24h", "7d"):
        rolled = grp["amount_gbp"].rolling(window, closed="left", min_periods=0)
        df[f"card_txn_count_{window}"] = (
            rolled.count().reset_index(level=0, drop=True).astype(int)
        )
        df[f"card_amount_sum_{window}"] = (
            rolled.sum().reset_index(level=0, drop=True).fillna(0.0)
        )
    return df


def _card_merchant_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card-per-merchant transaction count in the preceding 1 h.

    Groups by (card_id, merchant_id) so the count reflects how many times
    this specific card has hit this specific merchant recently. A legitimate
    one-off purchase has count=0; a card-testing ring probing the same merchant
    or a repeat fraudster returns to the same shop both produce elevated counts.

    This replaced the earlier merch_txn_count_1h (merchant-level total), which
    was semantically backwards: high-volume merchants were always elevated,
    while novel compromised merchants had low counts — exactly when fraud is
    highest. See todolist.md R4-R1.
    """
    rolled = df.groupby(["card_id", "merchant_id"], sort=False)["amount_gbp"].rolling(
        "1h", closed="left", min_periods=0
    )
    df["card_merch_txn_count_1h"] = (
        rolled.count().reset_index(level=[0, 1], drop=True).astype(int)
    )
    return df


def _time_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since this card's previous transaction; -1 for the first txn."""
    prev_ts = df.groupby("card_id")["timestamp_sec"].shift(1)
    df["time_since_last_card_txn_sec"] = (df["timestamp_sec"] - prev_ts).fillna(-1)
    return df


def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount relative to the card's 30-day baseline and log-scaled amount."""
    # clip at 0.01 (not 1.0) — preserves signal for low-spending cards (gift cards,
    # newly-issued cards) which are high-risk segments in CNP fraud.
    # NOTE: the deployed model was trained with clip=1.0. Retrain required before
    # deploying callers that use this updated clip value (see todolist.md).
    df["amount_to_card_avg_ratio"] = (
        df["amount_gbp"] / df["card_avg_amount_30d"].clip(lower=0.01)
    )
    df["log_amount"] = np.log1p(df["amount_gbp"])
    return df


def _cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encoding for hour and day-of-week. Periodicity preserved: hour 23 is adjacent to 0."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Accepts the raw DataFrame from src.data.generate (or its CSV equivalent)
    and returns a new DataFrame with all engineered columns appended.
    Original row order is restored via a positional reset at the end.

    Args:
        df: Raw transactions DataFrame.

    Returns:
        DataFrame with all engineered features; original columns preserved.
    """
    df = _to_datetime_index(df)
    df = _card_velocity(df)
    df = _card_merchant_velocity(df)
    df = _time_since_last(df)
    df = _amount_features(df)
    df = _cyclic_features(df)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    print(f"[velocity] Loading raw data from {_RAW_PATH} ...")
    raw = pd.read_csv(_RAW_PATH)
    print(f"[velocity] {len(raw):,} rows loaded. Running feature engineering ...")
    featured = add_features(raw)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(_OUT_PATH, index=False)
    print(f"[velocity] Saved {len(featured):,} rows to {_OUT_PATH}")
    print(f"[velocity] Columns: {list(featured.columns)}")
