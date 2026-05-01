"""
GNN model definitions for the Meteorological Variable Graph.

Three architectures are implemented:
  VariableGCN       : Graph Convolutional Network (GCNConv)
  VariableGAT       : Graph Attention Network (GATConv) with attention extraction
  VariableGraphSAGE : GraphSAGE (SAGEConv)

All models:
  - Accept node feature matrices x of shape [num_nodes, num_node_features]
  - Use global mean pooling over nodes to produce a graph-level representation
  - Output logits for 4 heat-risk classes
"""

import logging
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import (
        GCNConv, GATConv, SAGEConv, global_mean_pool
    )
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning(
        "PyTorch Geometric not available; GNN models cannot be defined. "
        "Install with: pip install torch-geometric"
    )


if _HAS_PYG:

    class VariableGCN(nn.Module):
        """
        Graph Convolutional Network over meteorological variables.

        Architecture:
          GCNConv -> ReLU -> Dropout
          GCNConv -> ReLU -> Dropout
          GCNConv -> ReLU
          GlobalMeanPool
          Linear -> ReLU -> Dropout -> Linear (logits)
        """

        def __init__(
            self,
            num_node_features: int,
            hidden_dim: int = 64,
            num_classes: int = 4,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.conv1 = GCNConv(num_node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
            self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, edge_index))

            x = global_mean_pool(x, batch)  # [batch_size, hidden_dim//2]
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)


    class VariableGAT(nn.Module):
        """
        Graph Attention Network over meteorological variables.
        Supports extraction of attention weights for explainability.

        Architecture:
          GATConv (8 heads) -> ELU -> Dropout
          GATConv (1 head)  -> ELU
          GlobalMeanPool
          Linear -> ELU -> Dropout -> Linear (logits)
        """

        def __init__(
            self,
            num_node_features: int,
            hidden_dim: int = 64,
            num_classes: int = 4,
            dropout: float = 0.3,
            heads: int = 4,
        ):
            super().__init__()
            self.conv1 = GATConv(
                num_node_features, hidden_dim // heads,
                heads=heads, dropout=dropout, concat=True,
            )
            self.conv2 = GATConv(
                hidden_dim, hidden_dim // 2,
                heads=1, dropout=dropout, concat=False,
            )
            self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
            self.dropout_layer = nn.Dropout(p=dropout)
            self._attention_weights = None  # storage for last attention

        def forward(self, x, edge_index, edge_attr=None, batch=None,
                    return_attention_weights: bool = False):
            # First GAT layer
            x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = self.dropout_layer(x)

            # Second GAT layer
            x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)

            # Store attention from second layer for explainability
            self._attention_weights = attn2

            x = global_mean_pool(x, batch)
            x = F.elu(self.fc1(x))
            x = self.dropout_layer(x)
            logits = self.fc2(x)

            if return_attention_weights:
                return logits, attn2
            return logits

        def get_attention_weights(self):
            """Return the attention weights from the most recent forward pass."""
            return self._attention_weights


    class VariableGraphSAGE(nn.Module):
        """
        GraphSAGE over meteorological variables.

        Architecture:
          SAGEConv -> ReLU -> Dropout
          SAGEConv -> ReLU -> Dropout
          SAGEConv -> ReLU
          GlobalMeanPool
          Linear -> ReLU -> Dropout -> Linear (logits)
        """

        def __init__(
            self,
            num_node_features: int,
            hidden_dim: int = 64,
            num_classes: int = 4,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.conv1 = SAGEConv(num_node_features, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
            self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, edge_index))

            x = global_mean_pool(x, batch)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)


    # ============================================================
    # MODEL FACTORY
    # ============================================================

    def build_gnn_model(
        model_name: str,
        num_node_features: int,
        hidden_dim: int = None,
        num_classes: int = None,
        dropout: float = None,
    ) -> nn.Module:
        """
        Instantiate a GNN model by name.

        Parameters
        ----------
        model_name       : 'GCN', 'GAT', or 'GraphSAGE'
        num_node_features: number of features per node
        hidden_dim       : hidden layer dimension (default from config)
        num_classes      : number of output classes (default from config)
        dropout          : dropout rate (default from config)

        Returns
        -------
        Instantiated (uninitialized-weights) model.
        """
        from src import config as cfg
        hd = hidden_dim or cfg.GNN_HIDDEN_DIM
        nc = num_classes or cfg.GNN_NUM_CLASSES
        dr = dropout or cfg.GNN_DROPOUT

        if model_name == "GCN":
            return VariableGCN(num_node_features, hd, nc, dr)
        elif model_name == "GAT":
            return VariableGAT(num_node_features, hd, nc, dr)
        elif model_name == "GraphSAGE":
            return VariableGraphSAGE(num_node_features, hd, nc, dr)
        else:
            raise ValueError(
                f"Unknown GNN model name '{model_name}'. "
                "Choose from: 'GCN', 'GAT', 'GraphSAGE'."
            )

else:
    # Provide stub so imports don't fail
    def build_gnn_model(*args, **kwargs):
        raise ImportError("PyTorch Geometric is not installed.")
