"""
Evaluation module for classical ML models.
Computes accuracy, balanced accuracy, macro F1, weighted F1, ROC-AUC,
confusion matrices, and classification reports.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

from src import config
from src.utils import save_dataframe, print_section
from src.visualization import plot_confusion_matrix_figure, plot_metric_comparison

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _compute_roc_auc(model, X, y_true):
    """Try to compute macro-OVR ROC-AUC; return NaN on failure."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return round(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro"
            ), 4)
    except Exception:
        pass
    return np.nan


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    target: str,
    save_cm: bool = True,
) -> dict:
    """
    Evaluate a single model and return a metrics dictionary.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "target": target,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "macro_precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc": _compute_roc_auc(model, X_test, y_test),
    }

    # Classification report
    report_str = classification_report(
        y_test, y_pred,
        target_names=[config.HEAT_RISK_CLASSES.get(i, str(i)) for i in range(4)],
        zero_division=0,
    )
    report_path = os.path.join(
        config.CLASSIFICATION_REPORTS_DIR,
        f"report_{model_name}_{target}.txt"
    )
    os.makedirs(config.CLASSIFICATION_REPORTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}  |  Target: {target}\n\n")
        f.write(report_str)

    # Confusion matrix
    if save_cm:
        plot_confusion_matrix_figure(
            y_test, y_pred, model_name=model_name, target=target,
            output_dir=config.CONFUSION_MATRICES_DIR,
        )

    return metrics


def evaluate_all_models(
    trained_models: dict,
    feature_info: dict,
) -> pd.DataFrame:
    """
    Evaluate all trained classical ML models on their respective test sets.

    Parameters
    ----------
    trained_models : {target: {model_name: model}}
    feature_info   : {target: {'X_test': ..., 'X_test_scaled': ..., 'y_test': ...}}

    Returns
    -------
    results_df : pd.DataFrame with one row per (model, target)
    """
    print_section("EVALUATING CLASSICAL ML MODELS")

    all_metrics = []

    for target, models_dict in trained_models.items():
        info = feature_info.get(target, {})
        y_test = info.get("y_test")
        X_test = info.get("X_test")
        X_test_scaled = info.get("X_test_scaled")

        if y_test is None or X_test is None:
            logger.warning(f"Feature info missing for target '{target}'; skipping evaluation.")
            continue

        for model_name, model in models_dict.items():
            Xte = X_test_scaled if model_name == "LogisticRegression" else X_test
            try:
                metrics = evaluate_model(model, Xte, y_test, model_name, target)
                all_metrics.append(metrics)
                print(f"  {model_name:20s} | {target:20s} | "
                      f"MacroF1={metrics['macro_f1']:.4f}  "
                      f"BalAcc={metrics['balanced_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}/{target}: {e}")

    results_df = pd.DataFrame(all_metrics)
    if results_df.empty:
        logger.warning("No evaluation results were produced.")
        return results_df

    save_dataframe(results_df, os.path.join(config.TABLES_DIR, "model_comparison.csv"))

    # Generate comparison plots
    plot_metric_comparison(
        results_df, "macro_f1",
        "macro_f1_comparison.png",
        "Macro F1-Score by Model and Horizon",
    )
    plot_metric_comparison(
        results_df, "balanced_accuracy",
        "balanced_accuracy_comparison.png",
        "Balanced Accuracy by Model and Horizon",
    )

    print(f"\n  Results saved -> {config.TABLES_DIR}/model_comparison.csv")
    return results_df


def find_best_model(results_df: pd.DataFrame) -> dict:
    """
    Identify the best classical ML model based on average Macro F1-score
    across all horizons. Prefers tree-based models for SHAP compatibility.

    Returns
    -------
    dict with keys: 'model_name', 'target', 'macro_f1'
    """
    if results_df.empty:
        return {}

    tree_models = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "DecisionTree"]
    tree_results = results_df[results_df["model"].isin(tree_models)]
    subset = tree_results if not tree_results.empty else results_df

    best_row = subset.loc[subset["macro_f1"].idxmax()]
    return {
        "model_name": best_row["model"],
        "target": best_row["target"],
        "macro_f1": best_row["macro_f1"],
    }
