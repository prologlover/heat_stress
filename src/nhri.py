"""
Nighttime Heat Recovery Index (NHRI) module.

NHRI_d = (number of nighttime hours with FeelsLikeC >= threshold)
          / (total nighttime hours in that day)

Two thresholds are computed: 35 °C and 40 °C.

Nighttime definition:
- If the 'isdaytime' column is available (yes/no or 0/1), use it.
- Otherwise use hour < 6 or hour >= 18.
"""

import os
import pandas as pd
import numpy as np
import logging

from src import config
from src.utils import save_dataframe, print_section

logger = logging.getLogger(__name__)


def _is_nighttime(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series: True where the row is a nighttime observation.
    """
    if "isdaytime" in df.columns:
        col = df["isdaytime"].astype(str).str.strip().str.lower()
        # Handle yes/no, true/false, 1/0
        day_vals = {"yes", "true", "1"}
        night_mask = ~col.isin(day_vals)
        logger.info("Using 'isdaytime' column to identify nighttime hours.")
        return night_mask

    if "hour" in df.columns:
        logger.info("Using hour (< 6 or >= 18) to identify nighttime hours.")
        return (df["hour"] < 6) | (df["hour"] >= 18)

    if "datetime" in df.columns:
        hour = pd.to_datetime(df["datetime"]).dt.hour
        logger.info("Deriving hour from 'datetime' to identify nighttime hours.")
        return (hour < 6) | (hour >= 18)

    raise ValueError(
        "Cannot determine nighttime: 'isdaytime', 'hour', and 'datetime' are all missing."
    )


def _nhri_category(nhri_val: float) -> str:
    """Return the qualitative NHRI category for a given value."""
    if nhri_val == 0.0:
        return "Full nighttime recovery"
    elif nhri_val <= 0.25:
        return "Low nighttime heat persistence"
    elif nhri_val <= 0.50:
        return "Moderate nighttime heat persistence"
    elif nhri_val <= 0.75:
        return "High nighttime heat persistence"
    else:
        return "Severe nighttime heat persistence"


def compute_nhri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily NHRI values for thresholds 35 °C and 40 °C.

    Parameters
    ----------
    df : hourly DataFrame with at least 'FeelsLikeC' and a date/datetime column.

    Returns
    -------
    daily_nhri : pd.DataFrame with one row per calendar date.
    """
    print_section("NIGHTTIME HEAT RECOVERY INDEX (NHRI)")

    if "FeelsLikeC" not in df.columns:
        raise ValueError("'FeelsLikeC' is required for NHRI computation.")

    out = df.copy()

    # Ensure we have a 'date' column (calendar date only)
    if "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"]).dt.date
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date
    else:
        raise ValueError("Neither 'datetime' nor 'date' column found.")

    # Ensure we have an 'hour' column for daytime stats
    if "hour" not in out.columns:
        if "datetime" in out.columns:
            out["hour"] = pd.to_datetime(out["datetime"]).dt.hour
        else:
            out["hour"] = 12  # fallback

    # Identify nighttime rows
    out["_is_night"] = _is_nighttime(out)

    # Build groups
    results = []
    for date, grp in out.groupby("date"):
        night = grp[grp["_is_night"]]
        day = grp[~grp["_is_night"]]

        total_night = len(night)
        feelslike_night = night["FeelsLikeC"]

        hot_35 = int((feelslike_night >= 35).sum())
        hot_40 = int((feelslike_night >= 40).sum())

        nhri_35 = hot_35 / total_night if total_night > 0 else np.nan
        nhri_40 = hot_40 / total_night if total_night > 0 else np.nan

        max_day_fl = day["FeelsLikeC"].max() if len(day) > 0 else np.nan
        mean_night_fl = feelslike_night.mean() if total_night > 0 else np.nan

        results.append({
            "date": date,
            "total_night_hours": total_night,
            "hot_night_hours_35": hot_35,
            "NHRI_35": round(nhri_35, 4) if not np.isnan(nhri_35) else np.nan,
            "hot_night_hours_40": hot_40,
            "NHRI_40": round(nhri_40, 4) if not np.isnan(nhri_40) else np.nan,
            "max_daytime_FeelsLikeC": round(max_day_fl, 2) if not np.isnan(max_day_fl) else np.nan,
            "mean_nighttime_FeelsLikeC": round(mean_night_fl, 2) if not np.isnan(mean_night_fl) else np.nan,
            "nhri_category_35": _nhri_category(nhri_35) if not np.isnan(nhri_35) else "N/A",
            "nhri_category_40": _nhri_category(nhri_40) if not np.isnan(nhri_40) else "N/A",
        })

    daily_nhri = pd.DataFrame(results)
    daily_nhri["date"] = pd.to_datetime(daily_nhri["date"])

    save_dataframe(daily_nhri, config.NHRI_DAILY_CSV)
    print(f"  Daily NHRI shape: {daily_nhri.shape}")

    # NHRI summary statistics
    summary_rows = []
    for thresh in [35, 40]:
        col = f"NHRI_{thresh}"
        cat_col = f"nhri_category_{thresh}"
        vals = daily_nhri[col].dropna()
        summary_rows.append({
            "threshold": thresh,
            "mean_nhri": round(vals.mean(), 4),
            "max_nhri": round(vals.max(), 4),
            "pct_days_nhri_gt_50": round(100 * (vals > 0.5).mean(), 2),
            "pct_days_full_recovery": round(100 * (vals == 0).mean(), 2),
            "most_common_category": daily_nhri[cat_col].mode()[0] if len(daily_nhri) > 0 else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, os.path.join(config.TABLES_DIR, "nhri_summary.csv"))
    print("  NHRI summary saved.")

    return daily_nhri


def merge_nhri_into_hourly(df: pd.DataFrame, daily_nhri: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily NHRI values back into the hourly dataset by calendar date.

    Parameters
    ----------
    df         : hourly DataFrame
    daily_nhri : daily NHRI DataFrame (output of compute_nhri)

    Returns
    -------
    Merged hourly DataFrame with NHRI columns added.
    """
    out = df.copy()

    # Ensure a 'date' column on the hourly frame
    if "datetime" in out.columns:
        out["_merge_date"] = pd.to_datetime(out["datetime"]).dt.date
    elif "date" in out.columns:
        out["_merge_date"] = pd.to_datetime(out["date"]).dt.date
    else:
        logger.warning("Cannot merge NHRI: no date/datetime column found.")
        return out

    # Prepare NHRI for merge
    nhri_cols = ["date", "NHRI_35", "NHRI_40", "nhri_category_35", "nhri_category_40"]
    nhri_merge = daily_nhri[nhri_cols].copy()
    nhri_merge["_merge_date"] = pd.to_datetime(nhri_merge["date"]).dt.date
    nhri_merge = nhri_merge.drop(columns=["date"])

    out = out.merge(nhri_merge, on="_merge_date", how="left")
    out = out.drop(columns=["_merge_date"])

    # Fill any NaN NHRI values (e.g. first/last days)
    out["NHRI_35"] = out["NHRI_35"].ffill().bfill().fillna(0.0)
    out["NHRI_40"] = out["NHRI_40"].ffill().bfill().fillna(0.0)
    out["nhri_category_35"] = out["nhri_category_35"].fillna("Full nighttime recovery")
    out["nhri_category_40"] = out["nhri_category_40"].fillna("Full nighttime recovery")

    save_dataframe(out, config.HOURLY_WITH_NHRI_CSV)
    print(f"  Saved hourly+NHRI dataset -> {config.HOURLY_WITH_NHRI_CSV}  shape: {out.shape}")
    return out
