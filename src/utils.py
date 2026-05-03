"""
Utility functions for the Heat-Stress Early Warning project.
Provides directory management, I/O helpers, and data-safety tools.
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd

from src import config

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY MANAGEMENT
# ============================================================

def create_directories():
    """Create all required output and processed data directories."""
    dirs = [
        config.PROCESSED_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.MODELS_DIR,
        config.SHAP_DIR,
        config.CONFUSION_MATRICES_DIR,
        config.CLASSIFICATION_REPORTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("All output directories are ready.")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}.")


# ============================================================
# FILE I/O
# ============================================================

def save_dataframe(df: pd.DataFrame, path: str, index: bool = False):
    """Save a DataFrame to CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Saved DataFrame ({len(df)} rows) -> {path}")


def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file with a clear error message if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[safe_read_csv] Required file not found: {path}\n"
            "Please ensure raw data files are placed in data/raw/."
        )
    df = pd.read_csv(path, **kwargs)
    logger.info(f"Loaded {path} -> shape {df.shape}")
    return df


def save_json(obj, path: str):
    """Serialize a Python object to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info(f"Saved JSON -> {path}")


def load_json(path: str):
    """Load a JSON file into a Python object."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_json] File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# COLUMN UTILITIES
# ============================================================

def get_available_columns(df: pd.DataFrame, candidate_columns: list) -> list:
    """
    Return the subset of candidate_columns that actually exist in df.
    Logs a warning for any missing candidates.
    """
    available = [c for c in candidate_columns if c in df.columns]
    missing = [c for c in candidate_columns if c not in df.columns]
    if missing:
        logger.warning(f"Columns not found in DataFrame (will be skipped): {missing}")
    return available


def check_required_columns(df: pd.DataFrame, required_columns: list):
    """
    Raise a ValueError if any required column is absent from df.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) missing from DataFrame: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# DATETIME UTILITIES
# ============================================================

def ensure_datetime_column(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """
    Ensure that `col` exists as a proper pandas datetime column.
    If the column is already present but not datetime, coerce it.
    Returns the modified DataFrame.
    """
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. Available columns: {list(df.columns)}"
        )
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
        n_bad = df[col].isna().sum()
        if n_bad > 0:
            logger.warning(
                f"{n_bad} rows had unparseable datetimes in column '{col}' and were set to NaT."
            )
    return df


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_section(title: str):
    """Print a clearly visible section header to the console."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
