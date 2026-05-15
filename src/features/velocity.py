"""
Velocity and behavioural feature engineering for CNP fraud detection.

These features capture the rapid-fire card activity that distinguishes
a fraudster burning through a compromised card from normal spending:
  - Rolling counts/sums over 1 h and 24 h per card
  - Merchant-level velocity over 1 h
  - Time elapsed since the card's previous transaction
  - Amount deviation from the card's 30-day baseline
  - Cyclic hour/day-of-week encodings

All rolling features use closed='left' so each row only sees transactions
that occurred strictly before it — zero future leakage.
"""

import numpy as np
import pandas as pd

_BASE_TS = pd.Timestamp("2020-01-01")


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp and replace the index with a monotonic DatetimeIndex."""
    df = df.sort_values("timestamp_sec").copy()
    df.index = _BASE_TS + pd.to_timedelta(df["timestamp_sec"], unit="s")
    df.index.name = None
    return df


def _card_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card rolling transaction count and amount sum for 1 h and 24 h."""
    grp = df.groupby("card_id", sort=False)
    for window in ("1h", "24h"):
        rolled = grp["amount_gbp"].rolling(window, closed="left", min_periods=0)
        df[f"card_txn_count_{window}"] = (
            rolled.count().reset_index(level=0, drop=True).astype(int)
        )
        df[f"card_amount_sum_{window}"] = (
            rolled.sum().reset_index(level=0, drop=True).fillna(0.0)
        )
    return df


def _merchant_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-merchant transaction count in the preceding 1 h."""
    rolled = df.groupby("merchant_id", sort=False)["amount_gbp"].rolling(
        "1h", closed="left", min_periods=0
    )
    df["merch_txn_count_1h"] = (
        rolled.count().reset_index(level=0, drop=True).astype(int)
    )
    return df


def _time_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since this card's previous transaction; -1 for the first txn."""
    prev_ts = df.groupby("card_id")["timestamp_sec"].shift(1)
    df["time_since_last_card_txn_sec"] = (df["timestamp_sec"] - prev_ts).fillna(-1)
    return df


def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount relative to the card's 30-day baseline and log-scaled amount."""
    df["amount_to_card_avg_ratio"] = (
        df["amount_gbp"] / df["card_avg_amount_30d"].clip(lower=1.0)
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
    df = _merchant_velocity(df)
    df = _time_since_last(df)
    df = _amount_features(df)
    df = _cyclic_features(df)
    return df.reset_index(drop=True)
