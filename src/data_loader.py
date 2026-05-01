"""
Data loading module.
Loads raw CSV files, performs initial quality checks, and saves a dataset summary.
"""

import pandas as pd
import logging
from src import config
from src.utils import safe_read_csv, save_dataframe, print_section

logger = logging.getLogger(__name__)


def _date_range_str(df: pd.DataFrame) -> str:
    """Try to extract a human-readable date range from common date columns."""
    for col in ["datetime", "date"]:
        if col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                valid = parsed.dropna()
                if len(valid) > 0:
                    return f"{valid.min().date()} to {valid.max().date()}"
            except Exception:
                pass
    return "N/A"


def _check_dataframe(df: pd.DataFrame, name: str):
    """Print shape, column names, missing values, and duplicates."""
    print(f"\n--- {name} ---")
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    missing = df.isnull().sum().sum()
    dups = df.duplicated().sum()
    print(f"  Missing vals: {missing}")
    print(f"  Duplicates  : {dups}")


def load_raw_data():
    """
    Load all three raw CSV files.

    Returns
    -------
    locations_df  : pd.DataFrame
    hourly_df     : pd.DataFrame
    daily_df      : pd.DataFrame
    """
    print_section("LOADING RAW DATA")

    locations_df = safe_read_csv(config.LOCATIONS_CSV)
    hourly_df = safe_read_csv(config.HOURLY_CSV, low_memory=False)
    daily_df = safe_read_csv(config.DAILY_CSV, low_memory=False)

    _check_dataframe(locations_df, "locations.csv")
    _check_dataframe(hourly_df, "weather_data_1hr.csv")
    _check_dataframe(daily_df, "weather_data_24hr.csv")

    # Build and save dataset summary
    summary_rows = []
    for name, df in [
        ("locations.csv", locations_df),
        ("weather_data_1hr.csv", hourly_df),
        ("weather_data_24hr.csv", daily_df),
    ]:
        summary_rows.append({
            "file": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isnull().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum()),
            "date_range": _date_range_str(df),
        })

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, config.TABLES_DIR + "/dataset_summary.csv")
    print("\nDataset summary saved.")

    return locations_df, hourly_df, daily_df
