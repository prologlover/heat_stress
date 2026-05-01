"""
Visualization module.
Generates publication-ready figures for EDA, model evaluation, and NHRI analysis.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src import config
from src.utils import print_section

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Consistent color palette for heat-risk classes
RISK_COLORS = {
    0: "#4CAF50",   # Normal     - green
    1: "#FFC107",   # Caution    - amber
    2: "#FF5722",   # Danger     - deep orange
    3: "#B71C1C",   # Extreme Danger - dark red
}
RISK_LABELS = {0: "Normal", 1: "Caution", 2: "Danger", 3: "Extreme Danger"}

try:
    plt.style.use(config.FIGURE_STYLE)
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")


def _save_fig(fig, filename: str):
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure -> {path}")


# ============================================================
# 1. HEAT-STRESS CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(df: pd.DataFrame):
    if "heat_risk_class" not in df.columns:
        logger.warning("'heat_risk_class' missing; skipping class distribution plot.")
        return
    counts = df["heat_risk_class"].value_counts().sort_index()
    labels = [RISK_LABELS.get(int(c), str(c)) for c in counts.index]
    colors = [RISK_COLORS.get(int(c), "gray") for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_title("Heat-Stress Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Heat-Risk Class", fontsize=12)
    ax.set_ylabel("Number of Hourly Observations", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_fig(fig, "class_distribution.png")


# ============================================================
# 2. MONTHLY AVERAGE FeelsLikeC
# ============================================================

def plot_monthly_feelslike(df: pd.DataFrame):
    if "FeelsLikeC" not in df.columns or "month" not in df.columns:
        logger.warning("Required columns missing; skipping monthly FeelsLikeC plot.")
        return
    monthly = df.groupby("month")["FeelsLikeC"].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, color="#E53935")
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color="#E53935")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.axhline(32, color="#FFC107", linestyle="--", linewidth=1.2, label="Caution (32°C)")
    ax.axhline(40, color="#FF5722", linestyle="--", linewidth=1.2, label="Danger (40°C)")
    ax.axhline(52, color="#B71C1C", linestyle="--", linewidth=1.2, label="Extreme Danger (52°C)")
    ax.set_title("Monthly Average Feels-Like Temperature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Mean FeelsLikeC (°C)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig(fig, "monthly_feelslike.png")


# ============================================================
# 3. YEARLY MAXIMUM FeelsLikeC
# ============================================================

def plot_yearly_max_feelslike(df: pd.DataFrame):
    if "FeelsLikeC" not in df.columns or "year" not in df.columns:
        logger.warning("Required columns missing; skipping yearly max FeelsLikeC plot.")
        return
    yearly = df.groupby("year")["FeelsLikeC"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly.index, yearly.values, color="#E53935", edgecolor="white")
    ax.set_title("Yearly Maximum Feels-Like Temperature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Maximum FeelsLikeC (°C)", fontsize=12)
    ax.axhline(52, color="#B71C1C", linestyle="--", linewidth=1.2, label="Extreme Danger (52°C)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(fig, "yearly_max_feelslike.png")


# ============================================================
# 4. MONTHLY FREQUENCY OF DANGER AND EXTREME DANGER
# ============================================================

def plot_monthly_danger_frequency(df: pd.DataFrame):
    if "heat_risk_class" not in df.columns or "month" not in df.columns:
        logger.warning("Required columns missing; skipping monthly danger frequency plot.")
        return
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    danger = df[df["heat_risk_class"].isin([2, 3])].groupby("month").size()
    danger = danger.reindex(range(1, 13), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(danger.index, danger.values, color="#FF5722", edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_title("Monthly Frequency of Danger and Extreme Danger Hours", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Number of Hours", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    _save_fig(fig, "monthly_danger_frequency.png")


# ============================================================
# 5. HOURLY FREQUENCY OF HEAT-STRESS CLASSES
# ============================================================

def plot_hourly_heat_risk_frequency(df: pd.DataFrame):
    if "heat_risk_class" not in df.columns or "hour" not in df.columns:
        logger.warning("Required columns missing; skipping hourly heat risk frequency plot.")
        return
    pivot = (
        df.groupby(["hour", "heat_risk_class"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [RISK_LABELS.get(int(c), str(c)) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [RISK_COLORS[i] for i in sorted(RISK_COLORS) if RISK_LABELS[i] in pivot.columns]
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Hourly Frequency of Heat-Stress Classes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Number of Observations", fontsize=12)
    ax.legend(title="Heat-Risk Class", fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_fig(fig, "hourly_heat_risk_frequency.png")


# ============================================================
# 6. DAY VS NIGHT HEAT-STRESS COMPARISON
# ============================================================

def plot_day_night_comparison(df: pd.DataFrame):
    if "heat_risk_class" not in df.columns:
        logger.warning("'heat_risk_class' missing; skipping day/night comparison plot.")
        return

    if "isdaytime" in df.columns:
        day_mask = df["isdaytime"].astype(str).str.lower().isin(["yes", "true", "1"])
    elif "hour" in df.columns:
        day_mask = (df["hour"] >= 6) & (df["hour"] < 18)
    else:
        logger.warning("Cannot determine daytime; skipping day/night comparison plot.")
        return

    day_dist = df[day_mask]["heat_risk_class"].value_counts(normalize=True).sort_index() * 100
    night_dist = df[~day_mask]["heat_risk_class"].value_counts(normalize=True).sort_index() * 100
    all_classes = sorted(set(day_dist.index) | set(night_dist.index))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_classes))
    width = 0.35
    day_vals = [day_dist.get(c, 0) for c in all_classes]
    night_vals = [night_dist.get(c, 0) for c in all_classes]

    ax.bar(x - width / 2, day_vals, width, label="Daytime", color="#FF9800", edgecolor="white")
    ax.bar(x + width / 2, night_vals, width, label="Nighttime", color="#3F51B5", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([RISK_LABELS.get(int(c), str(c)) for c in all_classes])
    ax.set_title("Day vs Night Heat-Stress Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Heat-Risk Class", fontsize=12)
    ax.set_ylabel("Percentage of Observations (%)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_fig(fig, "day_night_heat_comparison.png")


# ============================================================
# 7 & 8. YEARLY NHRI TREND
# ============================================================

def plot_yearly_nhri(df: pd.DataFrame, nhri_col: str = "NHRI_35"):
    if nhri_col not in df.columns or "year" not in df.columns:
        logger.warning(f"'{nhri_col}' or 'year' missing; skipping NHRI trend plot.")
        return
    yearly = df.groupby("year")[nhri_col].mean()
    thresh = nhri_col.split("_")[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, color="#7B1FA2")
    ax.fill_between(yearly.index, yearly.values, alpha=0.2, color="#7B1FA2")
    ax.set_title(f"Yearly Mean {nhri_col} (Threshold {thresh}°C)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(f"Mean {nhri_col}", fontsize=12)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_fig(fig, f"yearly_{nhri_col.lower()}.png")


# ============================================================
# 9. NHRI CATEGORY DISTRIBUTION
# ============================================================

def plot_nhri_category_distribution(df: pd.DataFrame):
    col = "nhri_category_35"
    if col not in df.columns:
        logger.warning(f"'{col}' missing; skipping NHRI category distribution plot.")
        return
    counts = df[col].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color="#7B1FA2", edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title("NHRI Category Distribution (Threshold 35°C)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Days", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, "nhri_category_distribution.png")


# ============================================================
# 10. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df: pd.DataFrame, max_features: int = 25):
    numeric = df.select_dtypes(include=[np.number])
    # Remove target and label columns
    drop_cols = [c for c in numeric.columns if c.startswith("risk_t_plus_")]
    drop_cols += ["heat_risk_class"]
    numeric = numeric.drop(columns=[c for c in drop_cols if c in numeric.columns])

    if numeric.shape[1] < 2:
        logger.warning("Not enough numeric columns for correlation heatmap.")
        return

    # Cap at max_features most-variant columns for readability
    if numeric.shape[1] > max_features:
        variances = numeric.var().sort_values(ascending=False)
        numeric = numeric[variances.index[:max_features]]

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    _save_fig(fig, "correlation_heatmap.png")


# ============================================================
# MODEL EVALUATION FIGURES
# ============================================================

def plot_metric_comparison(results_df: pd.DataFrame, metric: str, filename: str, title: str):
    """
    Bar chart comparing a metric across models and horizons.

    Parameters
    ----------
    results_df : DataFrame with columns ['model', 'target', metric]
    metric     : column name of the metric to plot
    filename   : output file name
    title      : plot title
    """
    if metric not in results_df.columns:
        logger.warning(f"Metric '{metric}' not found in results; skipping plot.")
        return

    pivot = results_df.pivot_table(index="model", columns="target", values=metric)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title="Horizon", fontsize=9, loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save_fig(fig, filename)


def plot_confusion_matrix_figure(
    y_true, y_pred, model_name: str, target: str,
    output_dir: str = None, prefix: str = ""
):
    """Save a confusion matrix figure."""
    if output_dir is None:
        output_dir = config.CONFUSION_MATRICES_DIR
    os.makedirs(output_dir, exist_ok=True)

    class_names = [RISK_LABELS.get(i, str(i)) for i in range(4)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(f"Confusion Matrix\n{model_name} | {target}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    safe_name = f"{prefix}{model_name}_{target}".replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"cm_{safe_name}.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix -> {path}")


# ============================================================
# MAIN EDA ENTRY POINT
# ============================================================

def generate_eda_figures(df: pd.DataFrame):
    """Run all EDA visualization functions."""
    print_section("GENERATING EDA FIGURES")
    plot_class_distribution(df)
    plot_monthly_feelslike(df)
    plot_yearly_max_feelslike(df)
    plot_monthly_danger_frequency(df)
    plot_hourly_heat_risk_frequency(df)
    plot_day_night_comparison(df)
    if "NHRI_35" in df.columns:
        plot_yearly_nhri(df, "NHRI_35")
        plot_yearly_nhri(df, "NHRI_40")
        plot_nhri_category_distribution(df)
    plot_correlation_heatmap(df)
    print("  All EDA figures saved to outputs/figures/")
