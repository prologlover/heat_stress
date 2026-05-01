"""
GNN evaluation module.
Computes the same metrics as classical ML evaluation for GNN models.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd

from src import config
from src.utils import save_dataframe, print_section
from src.visualization import plot_confusion_matrix_figure, plot_metric_comparison

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_GNN_AVAILABLE = False
try:
    import torch
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, classification_report,
    )
    _GNN_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available; GNN evaluation skipped.")


def evaluate_one_gnn(model, test_dataset, model_name: str, horizon: str) -> dict:
    """
    Evaluate one GNN model on its test dataset.

    Returns
    -------
    dict of evaluation metrics
    """
    if not _GNN_AVAILABLE:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=config.GNN_BATCH_SIZE, shuffle=False)
    all_preds, all_true, all_proba = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            proba = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch.y.cpu().numpy())
            all_proba.append(proba)

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_proba = np.vstack(all_proba) if all_proba else None

    # ROC-AUC
    roc_auc = np.nan
    try:
        if y_proba is not None:
            roc_auc = round(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"), 4)
    except Exception:
        pass

    metrics = {
        "model": model_name,
        "model_type": "GNN",
        "target": horizon,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc": roc_auc,
    }

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=[config.HEAT_RISK_CLASSES.get(i, str(i)) for i in range(4)],
        zero_division=0,
    )
    os.makedirs(config.GNN_CLASSIFICATION_REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(
        config.GNN_CLASSIFICATION_REPORTS_DIR,
        f"gnn_report_{model_name}_{horizon}.txt",
    )
    with open(report_path, "w") as f:
        f.write(f"GNN Model: {model_name}  |  Horizon: {horizon}\n\n")
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix_figure(
        y_true, y_pred,
        model_name=f"GNN_{model_name}",
        target=horizon,
        output_dir=config.GNN_CONFUSION_MATRICES_DIR,
        prefix="gnn_",
    )

    return metrics


def evaluate_all_gnn_models(trained_gnn: dict) -> pd.DataFrame:
    """
    Evaluate all trained GNN models.

    Parameters
    ----------
    trained_gnn : {horizon: {model_name: {'model': ..., 'test_ds': ..., ...}}}

    Returns
    -------
    gnn_results_df : DataFrame with one row per (GNN model, horizon)
    """
    print_section("EVALUATING GNN MODELS")

    if not _GNN_AVAILABLE:
        logger.warning("GNN evaluation skipped.")
        return pd.DataFrame()

    all_metrics = []

    for horizon, models_dict in trained_gnn.items():
        for model_name, info in models_dict.items():
            model = info.get("model")
            test_ds = info.get("test_ds")

            if model is None or test_ds is None:
                logger.warning(f"Missing model or test dataset for {model_name}/{horizon}.")
                continue

            try:
                metrics = evaluate_one_gnn(model, test_ds, model_name, horizon)
                all_metrics.append(metrics)
                print(f"  GNN {model_name:12s} | {horizon:20s} | "
                      f"MacroF1={metrics.get('macro_f1', 'N/A'):.4f}  "
                      f"BalAcc={metrics.get('balanced_accuracy', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"GNN evaluation failed for {model_name}/{horizon}: {e}")

    gnn_results_df = pd.DataFrame(all_metrics)
    if not gnn_results_df.empty:
        save_dataframe(
            gnn_results_df,
            os.path.join(config.TABLES_DIR, "gnn_model_comparison.csv"),
        )

    return gnn_results_df


def build_final_comparison_table(
    classical_results: pd.DataFrame,
    gnn_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine classical ML and GNN results into a unified comparison table.

    Returns
    -------
    final_df : pd.DataFrame
    """
    print_section("FINAL MODEL COMPARISON TABLE")

    frames = []

    if not classical_results.empty:
        cl = classical_results.copy()
        cl["model_type"] = "Classical ML"
        frames.append(cl)

    if not gnn_results.empty:
        frames.append(gnn_results)

    if not frames:
        logger.warning("No results available for final comparison table.")
        return pd.DataFrame()

    final_df = pd.concat(frames, ignore_index=True)

    # Ensure consistent column order
    desired_cols = [
        "model_type", "model", "target",
        "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "roc_auc",
    ]
    actual_cols = [c for c in desired_cols if c in final_df.columns]
    final_df = final_df[actual_cols]

    save_dataframe(
        final_df,
        os.path.join(config.TABLES_DIR, "final_model_comparison_all_models.csv"),
    )

    # Print summary
    if "macro_f1" in final_df.columns:
        best_row = final_df.loc[final_df["macro_f1"].idxmax()]
        print(f"\n  Best overall model:")
        print(f"    Type    : {best_row.get('model_type', 'N/A')}")
        print(f"    Model   : {best_row.get('model', 'N/A')}")
        print(f"    Target  : {best_row.get('target', 'N/A')}")
        print(f"    MacroF1 : {best_row.get('macro_f1', 'N/A'):.4f}")

    return final_df
