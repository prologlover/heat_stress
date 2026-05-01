"""
Meteorological Variable Graph Builder.

Builds a static graph G = (V, E) where:
  V = meteorological variables (nodes)
  E = relationships between variables (edges)

Three graph construction modes:
  'correlation' : edges based on Pearson correlation >= threshold
  'expert'      : edges based on meteorological domain knowledge
  'hybrid'      : union of correlation and expert edges (default)

For each timestamp, node features are the current value plus lag and
rolling statistics of each variable, forming a graph sample.
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
from src.utils import save_dataframe, get_available_columns, print_section

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logger.warning("NetworkX not installed; graph visualization will be skipped.")

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not installed; GNN dataset creation will be skipped.")

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning("PyTorch Geometric not installed; GNN dataset creation will be skipped.")


# ============================================================
# NODE FEATURE CONSTRUCTION
# ============================================================

# Lag and rolling windows used to build node feature vectors
_LAG_HOURS = [1, 3, 6, 24]
_ROLL_HOURS = [3, 6, 24]


def _build_node_feature_names(var: str) -> list:
    """Return the list of feature names for a single node variable."""
    names = [f"{var}_current"]
    for lag in _LAG_HOURS:
        names.append(f"{var}_lag_{lag}h")
    for win in _ROLL_HOURS:
        names.append(f"{var}_roll_{win}h")
    return names


def build_node_feature_matrix(df: pd.DataFrame, nodes: list) -> pd.DataFrame:
    """
    Construct the node-feature table for all timestamps.

    For each node variable, the feature vector is:
      [current, lag_1h, lag_3h, lag_6h, lag_24h, roll_3h, roll_6h, roll_24h]

    Parameters
    ----------
    df    : hourly DataFrame sorted chronologically
    nodes : list of node variable names (must exist in df)

    Returns
    -------
    feat_df : DataFrame with one column per (node, feature-type),
              row index aligned with df.
    """
    feat_dict = {}
    for var in nodes:
        feat_dict[f"{var}_current"] = df[var].values
        for lag in _LAG_HOURS:
            key = f"{var}_lag_{lag}h"
            if key in df.columns:
                feat_dict[key] = df[key].values
            else:
                feat_dict[key] = df[var].shift(lag).values
        for win in _ROLL_HOURS:
            key = f"{var}_roll_{win}h"
            if key in df.columns:
                feat_dict[key] = df[key].values
            else:
                feat_dict[key] = df[var].rolling(win, min_periods=1).mean().values

    return pd.DataFrame(feat_dict, index=df.index)


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_correlation_graph(train_df: pd.DataFrame, nodes: list, threshold: float = None):
    """
    Build edge list from Pearson correlation computed on the training set.

    Returns
    -------
    edges : list of (node_i, node_j, weight)
    """
    if threshold is None:
        threshold = config.GNN_CORRELATION_THRESHOLD

    avail = [n for n in nodes if n in train_df.columns]
    corr = train_df[avail].corr().abs()

    edges = []
    for i, ni in enumerate(avail):
        for j, nj in enumerate(avail):
            if j <= i:
                continue
            w = corr.loc[ni, nj]
            if w >= threshold:
                edges.append((ni, nj, round(float(w), 4)))
    logger.info(f"Correlation graph: {len(edges)} edges (threshold={threshold})")
    return edges


def build_expert_graph(nodes: list):
    """
    Build edge list from expert meteorological knowledge.

    Returns
    -------
    edges : list of (node_i, node_j, weight=1.0)
    """
    node_set = set(nodes)
    edges = []
    for (a, b) in config.GNN_EXPERT_EDGES:
        if a in node_set and b in node_set:
            edges.append((a, b, 1.0))
    logger.info(f"Expert graph: {len(edges)} edges")
    return edges


def build_hybrid_graph(train_df: pd.DataFrame, nodes: list, threshold: float = None):
    """
    Build hybrid graph combining expert and correlation edges.
    Duplicate edges are removed; expert edges take priority on weight.

    Returns
    -------
    edges : list of (node_i, node_j, weight)
    """
    corr_edges = build_correlation_graph(train_df, nodes, threshold)
    expert_edges = build_expert_graph(nodes)

    seen = set()
    merged = []
    for (a, b, w) in expert_edges + corr_edges:
        key = tuple(sorted([a, b]))
        if key not in seen:
            seen.add(key)
            merged.append((a, b, w))

    logger.info(f"Hybrid graph: {len(merged)} unique edges")
    return merged


def build_graph(train_df: pd.DataFrame, nodes: list, mode: str = None):
    """
    Dispatch graph construction based on mode.

    Parameters
    ----------
    mode : 'correlation', 'expert', or 'hybrid' (default from config)

    Returns
    -------
    nodes  : list of node names
    edges  : list of (node_i, node_j, weight)
    """
    if mode is None:
        mode = config.GNN_GRAPH_MODE

    if mode == "correlation":
        edges = build_correlation_graph(train_df, nodes)
    elif mode == "expert":
        edges = build_expert_graph(nodes)
    else:  # hybrid
        edges = build_hybrid_graph(train_df, nodes)

    return nodes, edges


# ============================================================
# GRAPH SAVING AND VISUALIZATION
# ============================================================

def save_graph_info(nodes: list, edges: list):
    """Save node list, edge list CSV and draw the graph."""
    node_df = pd.DataFrame({"node_id": range(len(nodes)), "variable": nodes})
    save_dataframe(node_df, os.path.join(config.TABLES_DIR, "gnn_node_list.csv"))

    edge_df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    save_dataframe(edge_df, os.path.join(config.TABLES_DIR, "gnn_edge_list.csv"))
    print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")

    if not _HAS_NX:
        return

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a, b, w) in edges:
        G.add_edge(a, b, weight=w)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=config.RANDOM_STATE, k=2.5)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#1565C0", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, width=[1 + 3 * w for w in edge_weights],
        edge_color="#FF7043", alpha=0.7, ax=ax,
    )
    ax.set_title(
        "Meteorological Variable Graph\n(Nodes = Variables, Edges = Relationships)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "meteorological_variable_graph.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved graph figure -> {path}")


# ============================================================
# PYTORCH GEOMETRIC DATASET
# ============================================================

if _HAS_TORCH and _HAS_PYG:

    class WeatherVariableGraphDataset(Dataset):
        """
        PyTorch Geometric dataset for the Meteorological Variable Graph.

        Each item is a torch_geometric.data.Data object:
            x          : [num_nodes, num_node_features] - dynamic node features
            edge_index : [2, num_edges] - graph connectivity
            edge_attr  : [num_edges, 1] - edge weights
            y          : scalar target label (heat-risk class)
        """

        def __init__(
            self,
            df: pd.DataFrame,
            nodes: list,
            edges: list,
            target_col: str,
            node_feat_df: pd.DataFrame = None,
            scaler=None,
        ):
            """
            Parameters
            ----------
            df           : hourly DataFrame (already split)
            nodes        : ordered list of node variable names
            edges        : list of (src_name, dst_name, weight)
            target_col   : name of the target column
            node_feat_df : pre-built node feature matrix (optional)
            scaler       : fitted StandardScaler for node features (optional)
            """
            super().__init__()
            self.nodes = nodes
            self.num_nodes = len(nodes)
            self.node_to_idx = {n: i for i, n in enumerate(nodes)}
            self.num_node_feats = 1 + len(_LAG_HOURS) + len(_ROLL_HOURS)  # 8

            # Build edge tensors
            src_idx = [self.node_to_idx[a] for (a, b, _) in edges if a in self.node_to_idx and b in self.node_to_idx]
            dst_idx = [self.node_to_idx[b] for (a, b, _) in edges if a in self.node_to_idx and b in self.node_to_idx]
            weights = [w for (a, b, w) in edges if a in self.node_to_idx and b in self.node_to_idx]

            # Add reverse edges (undirected)
            ei = torch.tensor(
                [src_idx + dst_idx, dst_idx + src_idx], dtype=torch.long
            )
            ew = torch.tensor(weights + weights, dtype=torch.float).unsqueeze(1)

            self.edge_index = ei
            self.edge_attr = ew

            # Node feature matrix: shape [T, num_nodes * num_node_feats]
            if node_feat_df is None:
                node_feat_df = build_node_feature_matrix(df, nodes)

            # Align index
            feat_array = node_feat_df.values.astype(np.float32)

            # Normalize
            if scaler is not None:
                feat_array = scaler.transform(feat_array)

            # Replace NaN with 0
            feat_array = np.nan_to_num(feat_array, nan=0.0)

            # Reshape to [T, num_nodes, num_node_feats]
            self.feat_array = feat_array.reshape(
                len(df), self.num_nodes, self.num_node_feats
            )

            # Targets
            self.targets = df[target_col].values.astype(np.int64)

            # Timestamps (optional metadata)
            if "datetime" in df.columns:
                self.timestamps = df["datetime"].values
            else:
                self.timestamps = np.arange(len(df))

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            x = torch.tensor(self.feat_array[idx], dtype=torch.float)
            y = torch.tensor(self.targets[idx], dtype=torch.long)
            return Data(
                x=x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=y,
            )


def create_gnn_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    nodes: list,
    edges: list,
    target_col: str,
) -> tuple:
    """
    Build train/val/test PyTorch Geometric datasets.

    Scaler is fitted on training features only.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset, scaler, node_feat_cols)
    """
    if not _HAS_TORCH or not _HAS_PYG:
        logger.warning("PyTorch or PyTorch Geometric not available; cannot create GNN datasets.")
        return None, None, None, None, None

    from sklearn.preprocessing import StandardScaler as SKScaler

    # Build feature matrices
    train_feat = build_node_feature_matrix(train_df, nodes)
    val_feat = build_node_feature_matrix(val_df, nodes)
    test_feat = build_node_feature_matrix(test_df, nodes)

    feat_cols = train_feat.columns.tolist()

    # Fill NaN before scaling
    train_feat = train_feat.fillna(0)
    val_feat = val_feat.fillna(0)
    test_feat = test_feat.fillna(0)

    # Fit scaler on training features only
    scaler = SKScaler()
    scaler.fit(train_feat.values)

    train_ds = WeatherVariableGraphDataset(
        train_df, nodes, edges, target_col,
        node_feat_df=train_feat, scaler=scaler,
    )
    val_ds = WeatherVariableGraphDataset(
        val_df, nodes, edges, target_col,
        node_feat_df=val_feat, scaler=scaler,
    )
    test_ds = WeatherVariableGraphDataset(
        test_df, nodes, edges, target_col,
        node_feat_df=test_feat, scaler=scaler,
    )

    logger.info(
        f"GNN datasets created: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, scaler, feat_cols
