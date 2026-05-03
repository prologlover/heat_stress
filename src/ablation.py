"""
Ablation study module.
Compares model performance across two experimental conditions:

  1. Classical ML without NHRI features
  2. Classical ML with NHRI features

Uses the same chronological split and evaluates using Macro F1 and Balanced Accuracy.
"""

import os
import warnings
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.utils import save_dataframe, print_section
from src.modeling import get_feature_columns, prepare_features
from src.evaluation import evaluate_model

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_NHRI_COLS = ["NHRI_35", "NHRI_40", "nhri_category_35", "nhri_category_40"]


# ============================================================
# CLASSICAL ML ABLATION (NHRI ON vs OFF)
# ============================================================

def _run_classical_ablation_condition(
    train_df, val_df, test_df,
    condition_name: str,
    target: str,
    exclude_nhri: bool = False,
    n_estimators: int = 100,
) -> dict:
    """Train RandomForest for one ablation condition and evaluate on test set."""
    from sklearn.ensemble import RandomForestClassifier

    feature_cols = get_feature_columns(train_df)
    if exclude_nhri:
        feature_cols = [c for c in feature_cols if c not in _NHRI_COLS]

    X_tr, X_va, X_te, _ = prepare_features(train_df, val_df, test_df, feature_cols)
    y_tr = train_df[target].astype(int)
    y_te = test_df[target].astype(int)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    metrics = evaluate_model(model, X_te, y_te, "RandomForest", target, save_cm=False)
    metrics["condition"] = condition_name
    metrics["model_type"] = "Classical ML"
    return metrics


# ============================================================
# PLOTS
# ============================================================

def _plot_ablation(ablation_df: pd.DataFrame, metric: str, title: str, filename: str):
    if ablation_df.empty or metric not in ablation_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1976D2", "#42A5F5"]
    bars = ax.bar(
        ablation_df["condition"],
        ablation_df[metric],
        color=colors[:len(ablation_df)],
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Experimental Condition", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ablation figure -> {path}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "risk_t_plus_1",
) -> pd.DataFrame:
    """
    Run the ablation study comparing Classical ML with and without NHRI features.

    Parameters
    ----------
    train_df, val_df, test_df : chronologically split DataFrames
    target                    : forecast horizon for ablation (default: 1-hour)

    Returns
    -------
    ablation_df : DataFrame with ablation results
    """
    print_section("ABLATION STUDY")
    n_est = config.QUICK_N_ESTIMATORS if config.QUICK_TEST_MODE else 100

    results = []

    # 1. Classical ML WITHOUT NHRI
    print("  [1/2] Classical ML without NHRI ...")
    try:
        m1 = _run_classical_ablation_condition(
            train_df, val_df, test_df,
            condition_name="Classical ML (no NHRI)",
            target=target,
            exclude_nhri=True,
            n_estimators=n_est,
        )
        results.append({
            "condition": m1["condition"],
            "model_type": m1["model_type"],
            "model": m1.get("model", "RandomForest"),
            "target": m1.get("target", target),
            "macro_f1": m1["macro_f1"],
            "balanced_accuracy": m1["balanced_accuracy"],
        })
    except Exception as e:
        logger.error(f"Condition 1 failed: {e}")

    # 2. Classical ML WITH NHRI
    print("  [2/2] Classical ML with NHRI ...")
    try:
        m2 = _run_classical_ablation_condition(
            train_df, val_df, test_df,
            condition_name="Classical ML (with NHRI)",
            target=target,
            exclude_nhri=False,
            n_estimators=n_est,
        )
        results.append({
            "condition": m2["condition"],
            "model_type": m2["model_type"],
            "model": m2.get("model", "RandomForest"),
            "target": m2.get("target", target),
            "macro_f1": m2["macro_f1"],
            "balanced_accuracy": m2["balanced_accuracy"],
        })
    except Exception as e:
        logger.error(f"Condition 2 failed: {e}")

    ablation_df = pd.DataFrame(results)
    if ablation_df.empty:
        logger.warning("Ablation study produced no results.")
        return ablation_df

    save_dataframe(ablation_df, os.path.join(config.TABLES_DIR, "ablation_study_results.csv"))

    _plot_ablation(
        ablation_df, "macro_f1",
        f"Ablation Study — Macro F1 (target: {target})",
        "ablation_macro_f1.png",
    )
    _plot_ablation(
        ablation_df, "balanced_accuracy",
        f"Ablation Study — Balanced Accuracy (target: {target})",
        "ablation_balanced_accuracy.png",
    )

    print("\n  Ablation results:")
    print(ablation_df[["condition", "macro_f1", "balanced_accuracy"]].to_string(index=False))
    return ablation_df
