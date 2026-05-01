"""
Feature engineering module.
Creates time features, cyclic encodings, lag features, rolling statistics,
and meteorological interaction proxies.
"""

import os
import pandas as pd
import numpy as np
import logging

from src import config
from src.utils import save_dataframe, print_section, get_available_columns

logger = logging.getLogger(__name__)

# Season mapping by month number
_SEASON_MAP = {
    12: 0, 1: 0, 2: 0,   # Winter
    3: 1, 4: 1, 5: 1,    # Spring
    6: 2, 7: 2, 8: 2,    # Summer
    9: 3, 10: 3, 11: 3,  # Autumn
}
_SEASON_LABELS = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar and cyclic time features from the datetime column."""
    if "datetime" not in df.columns:
        logger.warning("No 'datetime' column; time features skipped.")
        return df

    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["day_of_week"] = dt.dt.dayofweek      # 0 = Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map(_SEASON_MAP)
    return df


def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode periodic variables as sine/cosine pairs."""
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "winddirdegree" in df.columns:
        rad = np.deg2rad(df["winddirdegree"])
        df["winddir_sin"] = np.sin(rad)
        df["winddir_cos"] = np.cos(rad)

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for key meteorological variables."""
    lag_spec = {
        "tempC":      [1, 3, 6, 12, 24],
        "FeelsLikeC": [1, 3, 6, 12, 24],
        "humidity":   [1, 3, 6, 24],
    }
    for col, lags in lag_spec.items():
        if col not in df.columns:
            logger.warning(f"Lag features: column '{col}' not found; skipping.")
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling mean features for key meteorological variables."""
    roll_spec = {
        "tempC":      [3, 6, 12, 24],
        "FeelsLikeC": [3, 6, 12, 24],
        "humidity":   [3, 6, 24],
    }
    for col, windows in roll_spec.items():
        if col not in df.columns:
            logger.warning(f"Rolling features: column '{col}' not found; skipping.")
            continue
        for w in windows:
            df[f"{col}_roll_mean_{w}h"] = (
                df[col].rolling(window=w, min_periods=1).mean()
            )
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven interaction and proxy features."""
    interactions = [
        ("temperature_range_proxy",       "tempC",       "DewPointC",    "sub"),
        ("heat_humidity_interaction",      "tempC",       "humidity",     "mul"),
        ("wind_cooling_proxy",             "windspeedKmph","tempC",       "mul"),
        ("heatindex_humidity_interaction", "HeatIndexC",  "humidity",     "mul"),
        ("feelslike_wind_interaction",     "FeelsLikeC",  "windspeedKmph","mul"),
    ]
    for name, col_a, col_b, op in interactions:
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning(
                f"Interaction '{name}': column(s) '{col_a}' or '{col_b}' missing; skipping."
            )
            continue
        if op == "sub":
            df[name] = df[col_a] - df[col_b]
        elif op == "mul":
            df[name] = df[col_a] * df[col_b]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline.

    Parameters
    ----------
    df : labeled hourly DataFrame

    Returns
    -------
    DataFrame with engineered features.
    """
    print_section("FEATURE ENGINEERING")
    out = df.copy()

    out = _add_time_features(out)
    out = _add_cyclic_features(out)
    out = _add_lag_features(out)
    out = _add_rolling_features(out)
    out = _add_interaction_features(out)

    # Drop rows that have NaN in lag / rolling columns (caused by look-back)
    # Only drop NaN rows in feature columns, not in targets
    lag_roll_cols = [
        c for c in out.columns
        if ("_lag_" in c or "_roll_" in c)
    ]
    before = len(out)
    if lag_roll_cols:
        out = out.dropna(subset=lag_roll_cols).reset_index(drop=True)
    after = len(out)
    logger.info(
        f"Dropped {before - after} rows due to NaN in lag/rolling features. "
        f"Remaining: {after:,}"
    )

    print(f"  Feature-engineered shape: {out.shape}")
    print(f"  New columns added: {out.shape[1] - df.shape[1]}")

    save_dataframe(out, config.FEATURE_ENGINEERED_CSV)
    print(f"  Saved -> {config.FEATURE_ENGINEERED_CSV}")
    return out
