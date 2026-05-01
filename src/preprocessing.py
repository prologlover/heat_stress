"""
Preprocessing module.
Cleans raw hourly weather data and implements chronological train/val/test splitting.
"""

import os
import pandas as pd
import numpy as np
import logging

from src import config
from src.utils import save_dataframe, print_section, ensure_datetime_column

logger = logging.getLogger(__name__)

# Fahrenheit columns that are redundant when Celsius columns exist
_FAHRENHEIT_COLS = [
    "tempF", "FeelsLikeF", "HeatIndexF", "DewPointF",
    "WindChillF", "WindGustMiles", "visibilityMiles",
    "pressureInches", "precipInches", "windspeedMiles",
    "maxtempF", "mintempF", "avgtempF",
]

# Non-numeric / metadata columns to keep as-is (not forced to numeric)
_CATEGORICAL_KEEP = [
    "date", "time", "datetime", "isdaytime",
    "winddir16point", "weatherDesc", "weatherIconUrl",
    "moon_phase", "sunrise", "sunset", "moonrise", "moonset",
]


def _build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine date + time columns into a single 'datetime' column,
    or parse an existing datetime column.
    """
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    if "date" in df.columns and "time" in df.columns:
        # The 'time' column is stored as HHMM integer (e.g. 100 = 01:00, 1300 = 13:00)
        time_str = df["time"].astype(str).str.zfill(4)
        datetime_str = df["date"].astype(str) + " " + time_str.str[:2] + ":" + time_str.str[2:]
        df["datetime"] = pd.to_datetime(datetime_str, errors="coerce")
        return df

    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    raise ValueError(
        "Cannot construct datetime column: neither 'datetime', "
        "nor ('date' + 'time') columns are present."
    )


def _drop_redundant_fahrenheit(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Fahrenheit / imperial columns that have a Celsius equivalent."""
    cols_to_drop = [c for c in _FAHRENHEIT_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} redundant imperial columns.")
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely coerce columns to numeric where appropriate.
    Columns in _CATEGORICAL_KEEP are left unchanged.
    """
    for col in df.columns:
        if col in _CATEGORICAL_KEEP:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only replace if at least 70 % of non-null values converted successfully
            n_total = df[col].notna().sum()
            n_converted = converted.notna().sum()
            if n_total > 0 and (n_converted / n_total) >= 0.70:
                df[col] = converted
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numeric columns using time-series interpolation,
    with forward/backward fill as fallback.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "datetime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df = df.set_index("datetime")
        df[numeric_cols] = (
            df[numeric_cols]
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )
        df = df.reset_index()
    else:
        df[numeric_cols] = (
            df[numeric_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    return df


def preprocess_hourly_weather(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the hourly weather DataFrame.

    Steps:
    1. Build datetime column.
    2. Drop redundant Fahrenheit columns.
    3. Coerce object columns to numeric where appropriate.
    4. Sort chronologically.
    5. Remove duplicates.
    6. Interpolate / fill missing values.
    7. Validate that FeelsLikeC exists.
    8. Save cleaned dataset.

    Parameters
    ----------
    hourly_df : raw hourly weather DataFrame

    Returns
    -------
    Cleaned hourly DataFrame with a 'datetime' column.
    """
    print_section("PREPROCESSING HOURLY DATA")
    df = hourly_df.copy()

    # 1. Build datetime
    df = _build_datetime(df)
    n_bad_dt = df["datetime"].isna().sum()
    if n_bad_dt > 0:
        logger.warning(f"{n_bad_dt} rows have unparseable datetime and will be dropped.")
        df = df.dropna(subset=["datetime"])

    # 2. Drop redundant imperial columns
    df = _drop_redundant_fahrenheit(df)

    # 3. Coerce to numeric
    df = _coerce_numeric(df)

    # 4. Sort chronologically
    df = df.sort_values("datetime").reset_index(drop=True)

    # 5. Remove duplicates (keep last)
    before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.info(f"Removed {before - after} duplicate timestamps.")

    # 6. Fill missing values
    df = _fill_missing(df)

    # 7. Validate essential column
    if "FeelsLikeC" not in df.columns:
        raise ValueError(
            "Essential column 'FeelsLikeC' is missing from the dataset. "
            "Heat-stress labeling cannot proceed."
        )

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Date range   : {df['datetime'].min()} -> {df['datetime'].max()}")

    save_dataframe(df, config.CLEAN_HOURLY_CSV)
    print(f"  Saved -> {config.CLEAN_HOURLY_CSV}")
    return df


# ============================================================
# CHRONOLOGICAL SPLITTING
# ============================================================

def chronological_split(df: pd.DataFrame, datetime_col: str = "datetime"):
    """
    Split the DataFrame into training, validation, and test sets
    using the date boundaries defined in config.

    NO random shuffling is performed.

    Parameters
    ----------
    df          : DataFrame with a datetime column
    datetime_col: name of the datetime column

    Returns
    -------
    train_df, val_df, test_df
    """
    print_section("CHRONOLOGICAL SPLIT")

    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found for splitting.")

    dt = pd.to_datetime(df[datetime_col])

    train_mask = dt <= config.TRAIN_END
    val_mask = (dt >= config.VAL_START) & (dt <= config.VAL_END)
    test_mask = dt >= config.TEST_START

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    print(f"  Training   : {len(train_df):>7,} rows  "
          f"({train_df[datetime_col].min()} -> {train_df[datetime_col].max()})")
    print(f"  Validation : {len(val_df):>7,} rows  "
          f"({val_df[datetime_col].min()} -> {val_df[datetime_col].max()})")
    print(f"  Testing    : {len(test_df):>7,} rows  "
          f"({test_df[datetime_col].min()} -> {test_df[datetime_col].max()})")

    # Save split summary
    summary = pd.DataFrame([
        {
            "split": "train",
            "rows": len(train_df),
            "start": str(train_df[datetime_col].min()),
            "end": str(train_df[datetime_col].max()),
        },
        {
            "split": "validation",
            "rows": len(val_df),
            "start": str(val_df[datetime_col].min()),
            "end": str(val_df[datetime_col].max()),
        },
        {
            "split": "test",
            "rows": len(test_df),
            "start": str(test_df[datetime_col].min()),
            "end": str(test_df[datetime_col].max()),
        },
    ])
    save_dataframe(summary, os.path.join(config.TABLES_DIR, "split_summary.csv"))

    return train_df, val_df, test_df
