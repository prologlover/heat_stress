"""
SHAP explainability module for the best-performing tree-based classical ML model.
Falls back to permutation importance if SHAP fails.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.utils import save_dataframe, print_section

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.warning("SHAP not installed; will use permutation importance as fallback.")

# SHAP dependence candidate features
_DEPENDENCE_CANDIDATES = [
    "FeelsLikeC", "HeatIndexC", "humidity", "tempC",
    "DewPointC", "windspeedKmph", "hour",
    "NHRI_35", "NHRI_40",
]


def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure -> {path}")


def _normalize_shap_for_display(
    shap_values,
    n_classes_preference: int = 2,
):
    """
    TreeExplainer output shape varies by model, class count, and SHAP version.
    Return (shap_display, mean_abs_per_feature) with mean_abs strictly 1-D.
    """
    if isinstance(shap_values, list):
        arrs = [np.asarray(sv, dtype=float) for sv in shap_values]
        if not arrs:
            raise ValueError("Empty shap_values list")
        stack_abs = np.stack([np.abs(a) for a in arrs], axis=0)
        mean_abs = stack_abs.mean(axis=(0, 1))
        idx = min(n_classes_preference, len(arrs) - 1)
        shap_display = arrs[idx]
        return shap_display, np.ravel(mean_abs)

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 2:
        mean_abs = np.abs(arr).mean(axis=0)
        return arr, np.ravel(mean_abs)
    if arr.ndim == 3:
        # Typical: (n_samples, n_features, n_outputs)
        mean_abs = np.abs(arr).mean(axis=(0, 2))
        k = arr.shape[2]
        cls_idx = min(n_classes_preference, k - 1)
        return arr[:, :, cls_idx], np.ravel(mean_abs)

    raise ValueError(f"Unexpected shap_values ndim={arr.ndim}")


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def _run_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str):
    """
    Compute SHAP values and save summary, bar, and dependence plots.
    Returns a feature importance DataFrame or None on failure.
    """
    print(f"  Computing SHAP values for {model_name} ...")
    feature_names = X_test.columns.tolist()
    try:
        # Select explainer type
        tree_models = ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "DecisionTree")
        if model_name in tree_models:
            explainer = shap.TreeExplainer(model)
            # Numpy avoids pandas/shape edge cases inside TreeExplainer for some stacks.
            X_eval = np.ascontiguousarray(
                X_test.apply(pd.to_numeric, errors="coerce").astype(np.float64).values
            )
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
            X_eval = X_test

        shap_values = explainer.shap_values(X_eval)

        # For multiclass, shap_values is a list or (N, F, K) — normalize for tables/plots.
        shap_display, mean_abs_shap = _normalize_shap_for_display(shap_values)

        if mean_abs_shap.shape[0] != len(feature_names):
            raise ValueError(
                f"SHAP feature count {mean_abs_shap.shape[0]} != columns {len(feature_names)}"
            )

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        save_dataframe(
            importance_df,
            os.path.join(config.TABLES_DIR, "shap_feature_importance.csv"),
        )

        plot_X = (
            pd.DataFrame(X_eval, columns=feature_names)
            if model_name in tree_models
            else X_test
        )

        # SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_display, plot_X, feature_names=feature_names, show=False, max_display=20
        )
        plt.tight_layout()
        _save_fig(plt.gcf(), os.path.join(config.SHAP_DIR, "shap_summary.png"))

        # SHAP bar plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_display,
            plot_X,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        _save_fig(plt.gcf(), os.path.join(config.SHAP_DIR, "shap_bar.png"))

        # Dependence plots for candidate features
        for feat in _DEPENDENCE_CANDIDATES:
            if feat in X_test.columns:
                try:
                    fig3, ax3 = plt.subplots(figsize=(7, 5))
                    shap.dependence_plot(
                        feat, shap_display, plot_X, ax=ax3, show=False
                    )
                    plt.tight_layout()
                    _save_fig(fig3, os.path.join(config.SHAP_DIR, f"dependence_{feat}.png"))
                except Exception as dep_err:
                    logger.warning(f"Dependence plot failed for '{feat}': {dep_err}")

        print("  SHAP analysis complete.")
        return importance_df

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return None


# ============================================================
# PERMUTATION IMPORTANCE FALLBACK
# ============================================================

def _run_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Compute permutation feature importance as fallback when SHAP is unavailable.
    """
    from sklearn.inspection import permutation_importance
    print("  Computing permutation importance (SHAP fallback) ...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=config.RANDOM_STATE,
        scoring="f1_macro", n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns.tolist(),
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    save_dataframe(
        importance_df,
        os.path.join(config.TABLES_DIR, "shap_feature_importance.csv"),
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(importance_df) * 0.3)))
    top = importance_df.head(25)
    ax.barh(top["feature"][::-1], top["mean_importance"][::-1], color="#1976D2")
    ax.set_title("Permutation Feature Importance (SHAP Fallback)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Macro F1-Score", fontsize=11)
    plt.tight_layout()
    _save_fig(fig, os.path.join(config.FIGURES_DIR, "permutation_importance_fallback.png"))

    print("  Permutation importance complete.")
    return importance_df


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_shap_explainability(
    trained_models: dict,
    feature_info: dict,
    best_model_info: dict,
):
    """
    Run SHAP (or permutation importance) for the best tree-based model.

    Parameters
    ----------
    trained_models  : {target: {model_name: model}}
    feature_info    : {target: {X_train, X_test, y_test, ...}}
    best_model_info : {'model_name': ..., 'target': ...}
    """
    print_section("SHAP EXPLAINABILITY")

    if not best_model_info:
        logger.warning("No best model info available; skipping SHAP.")
        return

    model_name = best_model_info.get("model_name")
    target = best_model_info.get("target")

    print(f"  Best model: {model_name} | Target: {target}")

    model = trained_models.get(target, {}).get(model_name)
    if model is None:
        logger.warning(f"Model not found for {model_name}/{target}; skipping SHAP.")
        return

    info = feature_info.get(target, {})
    X_train = info.get("X_train")
    X_test = info.get("X_test")
    y_test = info.get("y_test")

    if X_test is None or y_test is None:
        logger.warning("Feature info incomplete; skipping SHAP.")
        return

    # Use scaled features for LogisticRegression
    if model_name == "LogisticRegression":
        X_train = info.get("X_train_scaled", X_train)
        X_test = info.get("X_test_scaled", X_test)

    if _HAS_SHAP:
        result = _run_shap(model, X_train, X_test, model_name)
        if result is None:
            logger.warning("SHAP failed; falling back to permutation importance.")
            _run_permutation_importance(model, X_test, y_test)
    else:
        _run_permutation_importance(model, X_test, y_test)
