"""
Ablation study module.
Compares model performance across five experimental conditions:

  1. Classical ML without NHRI features
  2. Classical ML with NHRI features
  3. GNN with correlation-only graph
  4. GNN with expert-only graph
  5. GNN with hybrid graph (default)

Uses the same chronological split and evaluates using Macro F1 and Balanced Accuracy.
"""

import os
import copy
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.utils import save_dataframe, print_section
from src.modeling import get_feature_columns, prepare_features, _build_models
from src.evaluation import evaluate_model

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_GNN_AVAILABLE = False
try:
    import torch
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import f1_score, balanced_accuracy_score
    _GNN_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available; GNN ablation will be skipped.")


# ============================================================
# CLASSICAL ML ABLATION (NHRI ON vs OFF)
# ============================================================

_NHRI_COLS = ["NHRI_35", "NHRI_40", "nhri_category_35", "nhri_category_40"]


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
# GNN ABLATION (GRAPH MODES)
# ============================================================

def _run_gnn_ablation_condition(
    train_df, val_df, test_df,
    nodes, graph_mode, condition_name, target,
) -> dict:
    """Train GCN with a specific graph mode and evaluate on the test set."""
    if not _GNN_AVAILABLE:
        return {}

    from src.graph_builder import build_graph, create_gnn_datasets
    from src.gnn_model import build_gnn_model
    from src.gnn_training import train_one_gnn

    _, edges = build_graph(train_df, nodes, mode=graph_mode)
    train_ds, val_ds, test_ds, _, _ = create_gnn_datasets(
        train_df, val_df, test_df, nodes, edges, target
    )

    if train_ds is None:
        return {}

    num_node_feats = 1 + 4 + 3  # current + 4 lags + 3 rolling
    model = build_gnn_model("GCN", num_node_features=num_node_feats)

    # Use shorter training for ablation
    ablation_epochs = 20 if not config.QUICK_TEST_MODE else 5
    trained_model, _ = train_one_gnn(
        model, train_ds, val_ds,
        model_name=f"GCN_ablation_{graph_mode}",
        horizon=target,
        max_epochs=ablation_epochs,
        patience=5,
    )

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trained_model.to(device)
    trained_model.eval()

    loader = DataLoader(test_ds, batch_size=config.GNN_BATCH_SIZE, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = trained_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)

    return {
        "condition": condition_name,
        "model_type": "GNN",
        "model": f"GCN ({graph_mode} graph)",
        "target": target,
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
    }


# ============================================================
# PLOTS
# ============================================================

def _plot_ablation(ablation_df: pd.DataFrame, metric: str, title: str, filename: str):
    if ablation_df.empty or metric not in ablation_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1976D2", "#42A5F5", "#E53935", "#FF7043", "#7B1FA2"]
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
    plt.xticks(rotation=25, ha="right", fontsize=9)
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
    nodes: list,
    target: str = "risk_t_plus_1",
) -> pd.DataFrame:
    """
    Run the ablation study comparing five experimental conditions.

    Parameters
    ----------
    train_df, val_df, test_df : chronologically split DataFrames
    nodes                     : list of GNN node variable names
    target                    : forecast horizon for ablation (default: 1-hour)

    Returns
    -------
    ablation_df : DataFrame with ablation results
    """
    print_section("ABLATION STUDY")
    n_est = config.QUICK_N_ESTIMATORS if config.QUICK_TEST_MODE else 100

    results = []
    conditions = []

    # 1. Classical ML WITHOUT NHRI
    print("  [1/5] Classical ML without NHRI ...")
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
            "model": m1["model"],
            "target": m1["target"],
            "macro_f1": m1["macro_f1"],
            "balanced_accuracy": m1["balanced_accuracy"],
        })
    except Exception as e:
        logger.error(f"Condition 1 failed: {e}")

    # 2. Classical ML WITH NHRI
    print("  [2/5] Classical ML with NHRI ...")
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
            "model": m2["model"],
            "target": m2["target"],
            "macro_f1": m2["macro_f1"],
            "balanced_accuracy": m2["balanced_accuracy"],
        })
    except Exception as e:
        logger.error(f"Condition 2 failed: {e}")

    # 3-5. GNN ablation over graph modes
    if _GNN_AVAILABLE:
        for idx, (gmode, cname) in enumerate([
            ("correlation", "GNN (correlation graph)"),
            ("expert",      "GNN (expert graph)"),
            ("hybrid",      "GNN (hybrid graph)"),
        ], start=3):
            print(f"  [{idx}/5] {cname} ...")
            try:
                gm = _run_gnn_ablation_condition(
                    train_df, val_df, test_df,
                    nodes=nodes, graph_mode=gmode,
                    condition_name=cname, target=target,
                )
                if gm:
                    results.append(gm)
            except Exception as e:
                logger.error(f"Condition {idx} failed: {e}")
    else:
        logger.warning("GNN ablation skipped (PyTorch Geometric not available).")

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
