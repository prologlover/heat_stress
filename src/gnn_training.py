"""
GNN training module with early stopping, class-weighted loss, and checkpoint saving.
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
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import f1_score
    _GNN_AVAILABLE = True
except ImportError:
    logger.warning(
        "PyTorch or PyTorch Geometric not available. "
        "GNN training will be skipped."
    )


def _compute_class_weights(targets: np.ndarray, num_classes: int = 4):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1e-6, counts)
    weights = counts.sum() / (num_classes * counts)
    return weights


def _save_curve(history: dict, key_y: str, key_y2: str, model_name: str, horizon: str):
    """Save a training curve figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history[key_y], label="Train", color="#1976D2")
    axes[0].plot(history["epoch"], history[f"val_{key_y}"], label="Validation", color="#E53935")
    axes[0].set_title(f"Loss — {model_name} | {horizon}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history[key_y2], label="Val Macro F1", color="#7B1FA2")
    axes[1].set_title(f"Val Macro F1 — {model_name} | {horizon}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()

    plt.tight_layout()
    safe = f"{model_name}_{horizon}".replace(" ", "_")
    loss_path = os.path.join(config.FIGURES_DIR, f"gnn_loss_curve_{safe}.png")
    fig.savefig(loss_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # Individual F1 curve figure
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(history["epoch"], history[key_y2], color="#7B1FA2", linewidth=2)
    ax2.set_title(f"Validation Macro F1 — {model_name} | {horizon}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    f1_path = os.path.join(config.FIGURES_DIR, f"gnn_val_f1_curve_{safe}.png")
    fig2.savefig(f1_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved training curves for {model_name}/{horizon}")


def train_one_gnn(
    model,
    train_dataset,
    val_dataset,
    model_name: str,
    horizon: str,
    max_epochs: int = None,
    patience: int = None,
    lr: float = None,
    batch_size: int = None,
):
    """
    Train one GNN model with early stopping.

    Parameters
    ----------
    model        : instantiated (untrained) GNN model
    train_dataset: WeatherVariableGraphDataset for training
    val_dataset  : WeatherVariableGraphDataset for validation
    model_name   : 'GCN', 'GAT', or 'GraphSAGE'
    horizon      : target column name (e.g. 'risk_t_plus_1')
    max_epochs   : maximum number of training epochs
    patience     : early-stopping patience
    lr           : learning rate
    batch_size   : mini-batch size

    Returns
    -------
    model : best trained model (loaded from checkpoint)
    history : dict with training metrics per epoch
    """
    if not _GNN_AVAILABLE:
        raise ImportError("PyTorch Geometric is not available.")

    max_epochs = max_epochs or (
        config.QUICK_GNN_MAX_EPOCHS if config.QUICK_TEST_MODE else config.GNN_MAX_EPOCHS
    )
    patience = patience or config.GNN_PATIENCE
    lr = lr or config.GNN_LR
    batch_size = batch_size or config.GNN_BATCH_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Class-weighted loss
    train_targets = np.array([int(train_dataset[i].y.item()) for i in range(len(train_dataset))])
    weights = _compute_class_weights(train_targets, num_classes=config.GNN_NUM_CLASSES)
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Torch version compatibility:
    # Some builds (especially in Colab / newer nightly mixes) do not accept
    # the `verbose` argument for ReduceLROnPlateau.
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    checkpoint_path = os.path.join(
        config.MODELS_DIR, f"gnn_{model_name}_{horizon}.pt"
    )

    history = {
        "epoch": [], "loss": [], "val_loss": [], "val_macro_f1": []
    }
    best_val_f1 = -1.0
    patience_counter = 0

    print(f"  Training {model_name} for {horizon} on {device} ...")

    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / max(len(train_dataset), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch.y.cpu().numpy())

        val_loss /= max(len(val_dataset), 1)
        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        scheduler.step(val_f1)

        history["epoch"].append(epoch)
        history["loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{max_epochs} | "
                  f"TrainLoss={train_loss:.4f} | "
                  f"ValLoss={val_loss:.4f} | "
                  f"ValMacroF1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(
        config.TABLES_DIR, f"gnn_training_history_{model_name}_{horizon}.csv"
    )
    save_dataframe(hist_df, hist_path)

    # Save curves
    _save_curve(history, "loss", "val_macro_f1", model_name, horizon)

    print(f"  Best val Macro F1 = {best_val_f1:.4f}")
    return model, history


def train_all_gnn_models(
    train_dataset_factory,   # callable(target_col) -> (train_ds, val_ds, test_ds, ...)
    nodes: list,
    edges: list,
) -> dict:
    """
    Train all GNN models for all horizons.

    Parameters
    ----------
    train_dataset_factory : function that accepts a target column name and returns
                            (train_ds, val_ds, test_ds)
    nodes                 : list of node variable names
    edges                 : list of edges

    Returns
    -------
    trained_gnn : {horizon: {model_name: model}}
    """
    if not _GNN_AVAILABLE:
        logger.warning("GNN training skipped: PyTorch Geometric not installed.")
        return {}

    print_section("TRAINING GNN MODELS")

    from src.gnn_model import build_gnn_model

    gnn_model_names = config.QUICK_GNN_MODELS if config.QUICK_TEST_MODE else config.GNN_MODELS
    horizons = config.QUICK_HORIZONS if config.QUICK_TEST_MODE else config.GNN_HORIZONS

    num_node_feats = len(nodes) * (1 + len([1, 3, 6, 24]) + len([3, 6, 24]))
    # Actually: each node has 8 features: [current + 4 lags + 3 rolling]
    # This is computed per-node: total columns = num_nodes * 8
    # But DataLoader collapses per graph, so num_node_features = 8 (per node)
    num_node_feats = 1 + 4 + 3  # 8

    trained_gnn = {}

    for horizon in horizons:
        print(f"\n  === Horizon: {horizon} ===")
        trained_gnn[horizon] = {}

        datasets = train_dataset_factory(horizon)
        if datasets is None or datasets[0] is None:
            logger.warning(f"Dataset factory returned None for {horizon}; skipping.")
            continue
        train_ds, val_ds, test_ds = datasets[0], datasets[1], datasets[2]

        for gnn_name in gnn_model_names:
            print(f"\n  Building {gnn_name} ...")
            try:
                model = build_gnn_model(
                    gnn_name, num_node_features=num_node_feats
                )
                trained_model, history = train_one_gnn(
                    model, train_ds, val_ds, gnn_name, horizon
                )
                trained_gnn[horizon][gnn_name] = {
                    "model": trained_model,
                    "test_ds": test_ds,
                    "history": history,
                }
            except Exception as e:
                logger.error(f"GNN training failed for {gnn_name}/{horizon}: {e}")

    return trained_gnn
