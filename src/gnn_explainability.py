"""
GNN explainability module.

Implements:
  A. GAT attention weight extraction (for VariableGAT)
  B. Permutation node importance (model-agnostic)
  C. Edge importance via masking
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

_GNN_AVAILABLE = False
try:
    import torch
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import f1_score
    _GNN_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available; GNN explainability will be skipped.")

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def _get_predictions(model, dataset, device):
    """Run inference and return (y_true, y_pred) numpy arrays."""
    loader = DataLoader(dataset, batch_size=config.GNN_BATCH_SIZE, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())
    return np.array(all_true), np.array(all_preds)


def _baseline_f1(model, dataset, device) -> float:
    y_true, y_pred = _get_predictions(model, dataset, device)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ============================================================
# A. GAT ATTENTION WEIGHTS
# ============================================================

def extract_gat_attention(model, test_dataset, nodes: list, edges: list) -> pd.DataFrame:
    """
    Extract and aggregate attention weights from VariableGAT.

    Returns
    -------
    attention_df : DataFrame with edge source, target, and mean attention weight
    """
    if not _GNN_AVAILABLE:
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=config.GNN_BATCH_SIZE, shuffle=False)

    all_edge_indices = []
    all_attentions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            try:
                out, attn = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    return_attention_weights=True,
                )
                ei, aw = attn  # edge_index, attention_weights
                all_edge_indices.append(ei.cpu().numpy())
                all_attentions.append(aw.cpu().numpy())
            except Exception as e:
                logger.warning(f"Attention extraction failed on batch: {e}")
                break

    if not all_attentions:
        logger.warning("No attention weights could be extracted.")
        return pd.DataFrame()

    # Use first batch as representative (graph structure is static)
    ei = all_edge_indices[0]  # [2, num_edges]
    aw = all_attentions[0]    # [num_edges, 1] or [num_edges, heads]

    if aw.ndim > 1:
        aw = aw.mean(axis=1)  # average over heads

    node_idx_to_name = {i: n for i, n in enumerate(nodes)}

    rows = []
    for k in range(ei.shape[1]):
        src = int(ei[0, k])
        dst = int(ei[1, k])
        rows.append({
            "source": node_idx_to_name.get(src, str(src)),
            "target": node_idx_to_name.get(dst, str(dst)),
            "attention_weight": round(float(aw[k]), 6),
        })

    attention_df = pd.DataFrame(rows)
    # Aggregate bidirectional edges
    attention_df = (
        attention_df
        .groupby(["source", "target"])["attention_weight"]
        .mean()
        .reset_index()
        .sort_values("attention_weight", ascending=False)
        .reset_index(drop=True)
    )
    return attention_df


# ============================================================
# B. PERMUTATION NODE IMPORTANCE
# ============================================================

def compute_node_importance(
    model,
    test_dataset,
    nodes: list,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Estimate node importance by permuting each node's features
    and measuring the drop in macro F1-score.

    Parameters
    ----------
    model       : trained GNN model
    test_dataset: WeatherVariableGraphDataset for the test split
    nodes       : ordered list of node names
    n_repeats   : number of permutation repeats

    Returns
    -------
    importance_df : DataFrame with columns [variable, importance_mean, importance_std]
    """
    if not _GNN_AVAILABLE:
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    baseline = _baseline_f1(model, test_dataset, device)
    print(f"  Baseline Macro F1: {baseline:.4f}")

    num_nodes = len(nodes)
    num_feats = test_dataset[0].x.shape[1]  # features per node

    results = []
    for node_idx, node_name in enumerate(nodes):
        drops = []
        for _ in range(n_repeats):
            # Build a permuted version of the dataset's x at node_idx
            loader = DataLoader(test_dataset, batch_size=config.GNN_BATCH_SIZE, shuffle=False)
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    x_perm = batch.x.clone()

                    # Identify node positions in the flattened batch
                    # batch.x shape: [total_nodes_in_batch, num_feats]
                    # For a single graph: nodes are 0..num_nodes-1
                    # DataLoader stacks graphs: node 'node_idx' repeats every num_nodes rows
                    num_graphs = batch.num_graphs
                    total_nodes = x_perm.shape[0]
                    nodes_per_graph = total_nodes // num_graphs

                    # Permute the features of node node_idx across all graphs in batch
                    node_positions = torch.arange(
                        node_idx, total_nodes, nodes_per_graph, device=device
                    )
                    if len(node_positions) > 0:
                        perm_order = torch.randperm(len(node_positions), device=device)
                        x_perm[node_positions] = x_perm[node_positions[perm_order]]

                    out = model(x_perm, batch.edge_index, batch.edge_attr, batch.batch)
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())
                    all_true.extend(batch.y.cpu().numpy())

            perm_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
            drops.append(baseline - perm_f1)

        results.append({
            "variable": node_name,
            "importance_mean": round(float(np.mean(drops)), 6),
            "importance_std": round(float(np.std(drops)), 6),
        })
        print(f"    {node_name:25s}: drop={np.mean(drops):.4f} ± {np.std(drops):.4f}")

    importance_df = pd.DataFrame(results).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    return importance_df


# ============================================================
# C. EDGE IMPORTANCE
# ============================================================

def compute_edge_importance(
    model,
    test_dataset,
    nodes: list,
    edges: list,
) -> pd.DataFrame:
    """
    Estimate edge importance by removing one edge at a time and measuring
    the drop in macro F1-score.

    Returns
    -------
    edge_imp_df : DataFrame with columns [source, target, importance]
    """
    if not _GNN_AVAILABLE:
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    baseline = _baseline_f1(model, test_dataset, device)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    results = []
    # Limit to first 20 edges for computational efficiency
    eval_edges = edges[:min(20, len(edges))]

    for (src_name, dst_name, weight) in eval_edges:
        src_idx = node_to_idx.get(src_name)
        dst_idx = node_to_idx.get(dst_name)
        if src_idx is None or dst_idx is None:
            continue

        loader = DataLoader(test_dataset, batch_size=config.GNN_BATCH_SIZE, shuffle=False)
        all_preds, all_true = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                ei = batch.edge_index
                ea = batch.edge_attr

                # Mask the target edge (both directions) across the batched graph
                num_graphs = batch.num_graphs
                nodes_per_graph = batch.x.shape[0] // num_graphs
                keep = torch.ones(ei.shape[1], dtype=torch.bool, device=device)

                for g in range(num_graphs):
                    offset = g * nodes_per_graph
                    g_src = src_idx + offset
                    g_dst = dst_idx + offset
                    # Remove both directed versions
                    for k in range(ei.shape[1]):
                        if (ei[0, k] == g_src and ei[1, k] == g_dst) or \
                           (ei[0, k] == g_dst and ei[1, k] == g_src):
                            keep[k] = False

                masked_ei = ei[:, keep]
                masked_ea = ea[keep] if ea is not None else None

                out = model(batch.x, masked_ei, masked_ea, batch.batch)
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
                all_true.extend(batch.y.cpu().numpy())

        masked_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        importance = baseline - masked_f1
        results.append({
            "source": src_name,
            "target": dst_name,
            "edge_weight": weight,
            "importance": round(float(importance), 6),
        })
        print(f"    ({src_name} -- {dst_name}): importance={importance:.4f}")

    edge_imp_df = pd.DataFrame(results).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    return edge_imp_df


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def _plot_importance(df: pd.DataFrame, value_col: str, label_col: str,
                     title: str, filename: str, color: str = "#1976D2"):
    if df.empty:
        return
    top = df.head(20)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
    ax.barh(top[label_col][::-1], top[value_col][::-1], color=color)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Drop in Macro F1)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved -> {path}")


def _plot_attention_graph(attention_df: pd.DataFrame, nodes: list):
    if attention_df.empty or not _HAS_NX:
        return
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for _, row in attention_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["attention_weight"])

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=config.RANDOM_STATE, k=2.5)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    normalized = [3 * w / max_w for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="#7B1FA2", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, width=normalized, edge_color="#FF9800",
        alpha=0.7, arrows=True, arrowsize=15, ax=ax,
    )
    ax.set_title("GAT Attention Graph\n(Edge thickness = attention weight)", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "gnn_attention_graph.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved attention graph -> {path}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_gnn_explainability(
    trained_gnn: dict,
    nodes: list,
    edges: list,
):
    """
    Run GNN explainability analysis.

    Picks the best-performing GNN (by val F1 history) and runs:
    - GAT attention extraction (if available)
    - Permutation node importance
    - Edge importance

    Parameters
    ----------
    trained_gnn : {horizon: {model_name: {'model': ..., 'test_ds': ..., 'history': ...}}}
    nodes       : list of node variable names
    edges       : list of (src, dst, weight) tuples
    """
    print_section("GNN EXPLAINABILITY")

    if not _GNN_AVAILABLE or not trained_gnn:
        logger.warning("GNN explainability skipped.")
        return

    # Select the horizon and model with the highest validation F1
    best_f1 = -1.0
    best_horizon, best_model_name, best_info = None, None, None

    for horizon, models_dict in trained_gnn.items():
        for model_name, info in models_dict.items():
            hist = info.get("history", {})
            val_f1s = hist.get("val_macro_f1", [0])
            peak_f1 = max(val_f1s) if val_f1s else 0
            if peak_f1 > best_f1:
                best_f1 = peak_f1
                best_horizon = horizon
                best_model_name = model_name
                best_info = info

    if best_info is None:
        logger.warning("No valid GNN model found for explainability.")
        return

    print(f"  Explaining {best_model_name} on {best_horizon} (val F1={best_f1:.4f})")
    model = best_info["model"]
    test_ds = best_info["test_ds"]

    # A. GAT attention (only for GAT model)
    attention_df = pd.DataFrame()
    if best_model_name == "GAT":
        print("  Extracting GAT attention weights ...")
        try:
            attention_df = extract_gat_attention(model, test_ds, nodes, edges)
            if not attention_df.empty:
                save_dataframe(
                    attention_df,
                    os.path.join(config.TABLES_DIR, "gnn_gat_attention_edges.csv"),
                )
                _plot_attention_graph(attention_df, nodes)
        except Exception as e:
            logger.error(f"GAT attention extraction failed: {e}")
    else:
        # Save empty placeholder
        pd.DataFrame(columns=["source", "target", "attention_weight"]).to_csv(
            os.path.join(config.TABLES_DIR, "gnn_gat_attention_edges.csv"), index=False
        )

    # B. Permutation node importance
    print("  Computing permutation node importance ...")
    try:
        node_imp_df = compute_node_importance(model, test_ds, nodes, n_repeats=5)
        if not node_imp_df.empty:
            save_dataframe(
                node_imp_df,
                os.path.join(config.TABLES_DIR, "gnn_node_importance.csv"),
            )
            _plot_importance(
                node_imp_df, "importance_mean", "variable",
                "GNN Node Importance (Permutation)", "gnn_node_importance.png",
                color="#1976D2",
            )
    except Exception as e:
        logger.error(f"Node importance computation failed: {e}")

    # C. Edge importance
    print("  Computing edge importance ...")
    try:
        edge_imp_df = compute_edge_importance(model, test_ds, nodes, edges)
        if not edge_imp_df.empty:
            edge_imp_df["edge_label"] = edge_imp_df["source"] + " -- " + edge_imp_df["target"]
            save_dataframe(
                edge_imp_df,
                os.path.join(config.TABLES_DIR, "gnn_edge_importance.csv"),
            )
            _plot_importance(
                edge_imp_df, "importance", "edge_label",
                "GNN Edge Importance (Masking)", "gnn_edge_importance.png",
                color="#E53935",
            )
    except Exception as e:
        logger.error(f"Edge importance computation failed: {e}")

    print("  GNN explainability complete.")
