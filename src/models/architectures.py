"""
Graph Neural Network Models for GRN Inference
==============================================

This module implements various GNN architectures for edge prediction.

We implement three models:
1. GCN (Graph Convolutional Network) - Simplest, good baseline
2. GAT (Graph Attention Network) - Learns attention weights
3. GraphSAGE - Sampling-based, scalable

All models follow the same pattern:
1. Encode nodes using GNN layers
2. Compute edge scores using learned node embeddings
3. Predict probability of regulatory interaction

Author: [Your Name]
Date: [Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, TransformerConv


def get_activation(name):
    """
    Factory function to get activation function by name.

    Args:
        name (str): Name of activation function

    Returns:
        callable: Activation function
    """
    activations = {
        'relu': F.relu,
        'leaky_relu': lambda x: F.leaky_relu(x, 0.2),
        'elu': F.elu,
        'gelu': F.gelu,
        'selu': F.selu,
        'silu': F.silu,  # Also known as Swish
        'mish': F.mish,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'softplus': F.softplus,
        'prelu': None,  # Requires nn.PReLU module, handled separately
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")

    return activations[name]


class EdgePredictor(nn.Module):
    """
    Edge prediction head for GRN inference.

    Given node embeddings from GNN, predicts probability that an edge exists.

    Uses rich edge features:
    - Concatenation: [z_i || z_j]
    - Difference: |z_i - z_j|
    - Hadamard product: z_i * z_j
    - Cosine similarity
    - Correlation (from data)

    Args:
        hidden_dim (int): Dimension of node embeddings
        dropout (float): Dropout rate
        activation (str): Activation function name
    """

    def __init__(self, hidden_dim, dropout=0.2, activation='relu'):
        super(EdgePredictor, self).__init__()

        # Input: 4*hidden_dim + 2 (concat, diff, product, cosine, correlation)
        self.fc1 = nn.Linear(4 * hidden_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.activation_name = activation

    def forward(self, z, edges, correlations, node_features=None):
        """
        Predict probabilities for given edges.

        Args:
            z (torch.Tensor): Node embeddings [num_nodes, hidden_dim]
            edges (torch.Tensor): Edge indices [2, num_edges]
            correlations (torch.Tensor): Correlation values for edges [num_edges]
            node_features: Unused, kept for API compatibility

        Returns:
            torch.Tensor: Edge logits [num_edges]
        """
        source_nodes = edges[0]
        target_nodes = edges[1]

        z_source = z[source_nodes]
        z_target = z[target_nodes]

        # Rich edge feature engineering
        concat = torch.cat([z_source, z_target], dim=1)
        diff = torch.abs(z_source - z_target)
        product = z_source * z_target
        cosine = F.cosine_similarity(z_source, z_target, dim=1, eps=1e-8).unsqueeze(1)
        corr_feature = correlations.unsqueeze(1)

        # Combine all features
        edge_features = torch.cat([concat, diff, product, cosine, corr_feature], dim=1)

        # Pass through MLP
        x = self.activation(self.fc1(edge_features))
        x = self.dropout(x)
        edge_logits = self.fc2(x).squeeze()

        return edge_logits


class GCN_GRN(nn.Module):
    """
    Graph Convolutional Network for GRN inference.

    GCN (Kipf & Welling 2017):
    - Aggregates neighbor features using normalized adjacency matrix
    - Simple and effective
    - Formula: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    Architecture:
    - Input: Node features
    - GCN layers: Learn node representations
    - Edge predictor: Predict regulatory edges

    Args:
        num_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GCN layers
        dropout (float): Dropout rate
        activation (str): Activation function name
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.2, activation='relu'):
        super(GCN_GRN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Define GCN layers
        # First layer: num_features -> hidden_dim
        # Middle layers: hidden_dim -> hidden_dim
        # Last layer: hidden_dim -> hidden_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Define edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout, activation)

    def forward(self, x, edge_index, edges_to_predict, correlations):
        """
        Forward pass of GCN model.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edges_to_predict (torch.Tensor): Edges to predict [2, num_pred_edges]

        Returns:
            torch.Tensor: Predicted edge probabilities [num_pred_edges]
        """
        # Store original features for edge predictor
        original_features = x

        # Apply GCN layers with configurable activation
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict edges with enhanced features
        edge_probs = self.edge_predictor(x, edges_to_predict, correlations, original_features)

        return edge_probs


class GAT_GRN(nn.Module):
    """
    Graph Attention Network for GRN inference.

    GAT (Veličković et al. 2018):
    - Learns attention weights for each neighbor
    - More expressive than GCN
    - Can identify most important regulatory relationships
    - Formula: α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))

    Benefits for GRN inference:
    - Attention weights show which gene relationships are important
    - Can handle varying numbers of regulators per gene
    - More interpretable than GCN

    Args:
        num_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GAT layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        activation (str): Activation function name (default: 'elu' for GAT)
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2,
                 num_heads=4, dropout=0.2, activation='elu'):
        super(GAT_GRN, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Define GAT layers
        # Multi-head attention in intermediate layers
        # Single-head attention in last layer (for simplicity)
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(
            num_features,
            hidden_dim,
            heads=num_heads,
            dropout=dropout
        ))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_dim * num_heads,  # Input is concatenated heads
                hidden_dim,
                heads=num_heads,
                dropout=dropout
            ))

        # Last layer (single head)
        self.convs.append(GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=1,
            dropout=dropout
        ))

        # Define edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout, activation)

    def forward(self, x, edge_index, edges_to_predict, correlations):
        """
        Forward pass of GAT model.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edges_to_predict (torch.Tensor): Edges to predict [2, num_pred_edges]

        Returns:
            torch.Tensor: Predicted edge probabilities [num_pred_edges]
        """
        # Store original features for edge predictor
        original_features = x

        # Apply GAT layers with configurable activation
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict edges with enhanced features
        edge_probs = self.edge_predictor(x, edges_to_predict, correlations, original_features)

        return edge_probs


class GraphSAGE_GRN(nn.Module):
    """
    GraphSAGE for GRN inference.

    GraphSAGE (Hamilton et al. 2017):
    - Samples neighbors instead of using all
    - Scalable to very large graphs
    - Learns aggregation function
    - Formula: h_i = σ(W · AGGREGATE({h_j, ∀j ∈ N(i)}))

    Benefits for large GRNs:
    - Handles graphs with >10,000 genes
    - Constant memory per node (samples fixed number of neighbors)
    - Can use mini-batch training

    Args:
        num_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of SAGE layers
        dropout (float): Dropout rate
        aggregator (str): Aggregation method ('mean', 'max', 'sum')
        activation (str): Activation function name
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2,
                 dropout=0.2, aggregator='mean', activation='relu'):
        super(GraphSAGE_GRN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Define GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggregator))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))

        # Define edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout, activation)

    def forward(self, x, edge_index, edges_to_predict, correlations):
        """
        Forward pass of GraphSAGE model.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edges_to_predict (torch.Tensor): Edges to predict [2, num_pred_edges]

        Returns:
            torch.Tensor: Predicted edge probabilities [num_pred_edges]
        """
        # Store original features for edge predictor
        original_features = x

        # Apply GraphSAGE layers with configurable activation
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict edges with enhanced features
        edge_probs = self.edge_predictor(x, edges_to_predict, correlations, original_features)

        return edge_probs


class MLP_GRN(nn.Module):
    """
    Multi-Layer Perceptron for GRN inference.

    MLP (Simple Neural Network):
    - Does NOT use graph structure (no message passing)
    - Encodes each node independently using MLP layers
    - Simpler and faster than GNN models
    - Good baseline for comparison

    Architecture:
    - Input: Node features
    - MLP layers: Learn node representations independently with residual connections
    - Edge predictor: Predict regulatory edges

    Args:
        num_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of MLP layers
        dropout (float): Dropout rate
        activation (str): Activation function name
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.2, activation='leaky_relu'):
        super(MLP_GRN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Define MLP layers for node encoding with increasing then decreasing dimensions
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer - expand dimensions
        self.layers.append(nn.Linear(num_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers - maintain dimension
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout, activation)

    def forward(self, x, edge_index, edges_to_predict, correlations):
        """
        Forward pass of MLP model with residual connections.

        Note: edge_index is ignored since MLP doesn't use graph structure.
        We keep it in the signature for consistency with GNN models.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges] (ignored)
            edges_to_predict (torch.Tensor): Edges to predict [2, num_pred_edges]

        Returns:
            torch.Tensor: Predicted edge probabilities [num_pred_edges]
        """
        # Store original features for edge predictor
        original_features = x

        # Apply MLP layers to encode nodes with batch norm and residual connections
        identity = None
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            if i == 0:
                x = layer(x)
                x = bn(x)
                x = self.activation(x)
                identity = x
            else:
                residual = x
                x = layer(x)
                x = bn(x)
                x = self.activation(x)
                # Add residual connection
                x = x + residual

            x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict edges using encoded node representations with enhanced features
        edge_logits = self.edge_predictor(x, edges_to_predict, correlations, original_features)

        return edge_logits


class GIN_GRN(nn.Module):
    """
    Graph Isomorphism Network for GRN inference.

    GIN (Xu et al. 2019):
    - Provably as powerful as the Weisfeiler-Lehman graph isomorphism test
    - Uses MLP for aggregation instead of simple sum/mean
    - Formula: h_i^(k) = MLP^(k)((1 + ε^(k)) · h_i^(k-1) + Σ_{j∈N(i)} h_j^(k-1))

    Benefits for GRN inference:
    - More expressive than GCN/GAT in distinguishing graph structures
    - Can capture subtle regulatory patterns
    - Works well for detecting structural motifs

    Args:
        num_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GIN layers
        dropout (float): Dropout rate
        activation (str): Activation function name
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2,
                 dropout=0.2, activation='relu'):
        super(GIN_GRN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Define GIN layers
        # Each GIN layer uses an MLP as the aggregation function
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),  # Internal MLP activation
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp1, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Define edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout, activation)

    def forward(self, x, edge_index, edges_to_predict, correlations):
        """
        Forward pass of GIN model.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edges_to_predict (torch.Tensor): Edges to predict [2, num_pred_edges]

        Returns:
            torch.Tensor: Predicted edge probabilities [num_pred_edges]
        """
        # Store original features for edge predictor
        original_features = x

        # Apply GIN layers with configurable activation
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict edges with enhanced features
        edge_probs = self.edge_predictor(x, edges_to_predict, correlations, original_features)

        return edge_probs


class GraphTransformer_GRN(nn.Module):
    """
    Graph Transformer for GRN inference.

    TransformerConv (Shi et al. 2020):
    - Uses multi-head attention mechanism from Transformers on graphs
    - Computes attention between connected nodes
    - More flexible attention mechanism than GAT

    Benefits for GRN inference:
    - State-of-the-art performance on many graph tasks
    - Better at capturing long-range dependencies
    - More expressive attention mechanism

    Args:
        num_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        activation (str): Activation function name
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2,
                 num_heads=4, dropout=0.2, activation='relu'):
        super(GraphTransformer_GRN, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Calculate output dimension per head
        # To ensure total output = hidden_dim, we need head_dim * num_heads = hidden_dim
        # If hidden_dim is not divisible by num_heads, we adjust
        head_dim = hidden_dim // num_heads
        actual_hidden_dim = head_dim * num_heads  # This will equal hidden_dim if divisible

        # If hidden_dim is not divisible by num_heads, adjust num_heads
        if actual_hidden_dim != hidden_dim:
            # Find a suitable number of heads that divides hidden_dim
            for h in [8, 4, 2, 1]:
                if hidden_dim % h == 0:
                    num_heads = h
                    head_dim = hidden_dim // num_heads
                    break
            self.num_heads = num_heads

        # Input projection layer to map num_features -> hidden_dim
        # This ensures consistent dimensions regardless of input feature size
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Define Transformer layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # All layers now take hidden_dim as input (after projection)
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                hidden_dim,
                head_dim,  # Output per head (head_dim * num_heads = hidden_dim)
                heads=num_heads,
                dropout=dropout,
                concat=True  # Concatenate heads -> outputs hidden_dim
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Define edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, dropout, activation)

    def forward(self, x, edge_index, edges_to_predict, correlations):
        """
        Forward pass of Graph Transformer model.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            edges_to_predict (torch.Tensor): Edges to predict [2, num_pred_edges]

        Returns:
            torch.Tensor: Predicted edge probabilities [num_pred_edges]
        """
        # Store original features for edge predictor
        original_features = x

        # Project input features to hidden_dim
        x = self.input_proj(x)
        x = self.activation(x)

        # Apply Transformer layers with configurable activation
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict edges with enhanced features
        edge_probs = self.edge_predictor(x, edges_to_predict, correlations, original_features)

        return edge_probs


def create_model(config, num_features):
    """
    Factory function to create model based on config.

    Supported model types:
    - 'mlp': Multi-Layer Perceptron (baseline, no graph structure)
    - 'gcn': Graph Convolutional Network
    - 'gat': Graph Attention Network
    - 'graphsage': GraphSAGE (sampling-based)
    - 'gin': Graph Isomorphism Network
    - 'transformer': Graph Transformer

    Supported activation functions:
    - 'relu', 'leaky_relu', 'elu', 'gelu', 'selu', 'silu', 'mish', 'tanh', 'softplus'

    Args:
        config (dict): Configuration dictionary
        num_features (int): Number of input node features

    Returns:
        nn.Module: GNN model
    """
    model_type = config['model']['model_type']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    activation = config['model'].get('activation', 'relu')

    # Create model based on type
    if model_type == 'mlp':
        model = MLP_GRN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
    elif model_type == 'gcn':
        model = GCN_GRN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
    elif model_type == 'gat':
        # GAT defaults to ELU if not specified
        gat_activation = activation if activation != 'relu' else 'elu'
        model = GAT_GRN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config['model'].get('num_heads', 4),
            dropout=dropout,
            activation=gat_activation
        )
    elif model_type == 'graphsage':
        aggregator = config['model'].get('aggregator', 'mean')
        model = GraphSAGE_GRN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggregator=aggregator,
            activation=activation
        )
    elif model_type == 'gin':
        model = GIN_GRN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
    elif model_type == 'transformer':
        model = GraphTransformer_GRN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config['model'].get('num_heads', 4),
            dropout=dropout,
            activation=activation
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: mlp, gcn, gat, graphsage, gin, transformer"
        )

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nCreated {model_type.upper()} model")
    print(f"  Parameters: {num_params:,}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Activation: {activation}")
    print(f"  Dropout: {dropout}")

    return model


# Available model types for reference
AVAILABLE_MODELS = ['mlp', 'gcn', 'gat', 'graphsage', 'gin', 'transformer']

# Available activation functions for reference
AVAILABLE_ACTIVATIONS = [
    'relu', 'leaky_relu', 'elu', 'gelu', 'selu', 'silu', 'mish', 'tanh', 'softplus'
]