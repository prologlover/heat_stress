"""
Heat-stress labeling module.
Assigns heat-risk classes based on FeelsLikeC and creates future prediction targets.
"""

import os
import pandas as pd
import numpy as np
import logging

from src import config
from src.utils import save_dataframe, print_section

logger = logging.getLogger(__name__)


def assign_heat_risk_class(feelslike: pd.Series) -> pd.Series:
    """
    Map FeelsLikeC values to integer heat-risk classes.

    Class 0 - Normal        : FeelsLikeC < 32
    Class 1 - Caution       : 32 <= FeelsLikeC < 40
    Class 2 - Danger        : 40 <= FeelsLikeC < 52
    Class 3 - Extreme Danger: FeelsLikeC >= 52
    """
    conditions = [
        feelslike < 32,
        (feelslike >= 32) & (feelslike < 40),
        (feelslike >= 40) & (feelslike < 52),
        feelslike >= 52,
    ]
    choices = [0, 1, 2, 3]
    return pd.Series(
        np.select(conditions, choices, default=np.nan),
        index=feelslike.index,
        name="heat_risk_class",
    )


def create_heat_stress_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add heat_risk_class and heat_risk_label columns, create future-horizon
    target columns, and save distribution summaries.

    Parameters
    ----------
    df : DataFrame with 'FeelsLikeC' column

    Returns
    -------
    DataFrame with new label and target columns.
    """
    print_section("HEAT-STRESS LABELING")

    if "FeelsLikeC" not in df.columns:
        raise ValueError(
            "Column 'FeelsLikeC' is required for heat-stress labeling but is missing."
        )

    out = df.copy()

    # Assign current heat-risk class and label
    out["heat_risk_class"] = assign_heat_risk_class(out["FeelsLikeC"]).astype("Int64")
    out["heat_risk_label"] = out["heat_risk_class"].map(config.HEAT_RISK_CLASSES)

    # Print class distribution
    dist = out["heat_risk_class"].value_counts().sort_index()
    print("\n  Current heat-risk class distribution:")
    for cls, count in dist.items():
        label = config.HEAT_RISK_CLASSES.get(int(cls), "Unknown")
        pct = 100 * count / len(out)
        print(f"    Class {cls} ({label:>14s}): {count:>7,} ({pct:5.1f}%)")

    # Save current distribution
    dist_df = pd.DataFrame({
        "class": dist.index.astype(int),
        "label": [config.HEAT_RISK_CLASSES.get(int(c), "Unknown") for c in dist.index],
        "count": dist.values,
        "pct": (100 * dist.values / len(out)).round(2),
    })
    save_dataframe(dist_df, os.path.join(config.TABLES_DIR, "heat_risk_distribution.csv"))

    # Create future target columns (shift by horizon hours)
    print("\n  Creating future target columns ...")
    future_dist_rows = []

    for h in config.FORECAST_HORIZONS:
        target_col = f"risk_t_plus_{h}"
        out[target_col] = out["heat_risk_class"].shift(-h)
        dist_h = out[target_col].value_counts().sort_index()
        for cls, count in dist_h.items():
            future_dist_rows.append({
                "horizon_hours": h,
                "target": target_col,
                "class": int(cls),
                "label": config.HEAT_RISK_CLASSES.get(int(cls), "Unknown"),
                "count": int(count),
                "pct": round(100 * count / dist_h.sum(), 2),
            })
        print(f"    {target_col}: {dist_h.sum():,} non-null rows")

    # Drop rows where ANY future target is NaN
    target_cols = config.TARGET_COLUMNS
    before = len(out)
    out = out.dropna(subset=target_cols).reset_index(drop=True)
    after = len(out)
    logger.info(f"Dropped {before - after} rows with NaN future targets.")

    # Cast targets to integer
    for col in target_cols:
        out[col] = out[col].astype(int)

    future_dist_df = pd.DataFrame(future_dist_rows)
    save_dataframe(
        future_dist_df,
        os.path.join(config.TABLES_DIR, "future_target_distributions.csv"),
    )

    print(f"\n  Final labeled dataset shape: {out.shape}")
    return out
