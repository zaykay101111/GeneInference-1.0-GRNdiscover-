import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import yaml
import argparse
from pathlib import Path
import pickle


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_expression_data(filepath, max_genes=None, tf_list=None):
    """
    Load gene expression matrix from CSV file.

    Expected format:
    - Rows: Genes
    - Columns: Cells
    - Index: Gene names
    - Header: Cell IDs

    Args:
        filepath (str): Path to expression data CSV
        max_genes (int): Maximum number of genes to use (for computational efficiency)
        tf_list (list): List of TF gene names to force-include in the gene set.
            If provided, TFs present in the expression data are always kept,
            even if they are not among the top variance genes.

    Returns:
        pd.DataFrame: Expression matrix (genes x cells)
    """
    print(f"\nLoading expression data from {filepath}")

    # load CSV file with gene names as index
    expression_data = pd.read_csv(filepath, index_col=0)

    if max_genes is not None and expression_data.shape[0] > max_genes:
        # compute variance for each gene across cells
        gene_var = expression_data.var(axis=1)

        if tf_list is not None:
            # Force-include TFs that exist in the expression data
            tfs_in_data = [tf for tf in tf_list if tf in expression_data.index]
            non_tf_genes = [g for g in expression_data.index if g not in tfs_in_data]

            # How many non-TF slots do we have?
            num_tf_included = len(tfs_in_data)
            num_non_tf_slots = max(0, max_genes - num_tf_included)

            # Select top variance non-TF genes to fill remaining slots
            non_tf_var = gene_var.loc[non_tf_genes]
            top_non_tf = non_tf_var.nlargest(num_non_tf_slots).index.tolist()

            # Combine: all TFs + top variance non-TFs
            selected_genes = tfs_in_data + top_non_tf
            expression_data = expression_data.loc[selected_genes]
            print(f"  Force-included {num_tf_included} TFs in gene set")
            print(f"  Added {len(top_non_tf)} top-variance non-TF genes")
        else:
            # No TF list: just take top variance genes
            top_genes = gene_var.nlargest(max_genes).index
            expression_data = expression_data.loc[top_genes]

    print(f"  Genes: {expression_data.shape[0]}")
    print(f"  Cells: {expression_data.shape[1]}")
    print(f"  Expression range: [{expression_data.min().min():.2f}, {expression_data.max().max():.2f}]")

    return expression_data


def normalize_expression(expression_data, method='log_zscore'):
    """
    Normalize gene expression values.

    Args:
        expression_data (pd.DataFrame): Raw expression matrix
        method (str): Normalization method

    Returns:
        pd.DataFrame: Normalized expression matrix
    """
    print(f"\nNormalizing expression data using {method}")

    if method == 'log':
        normalized = np.log(expression_data + 1)
    elif method == 'zscore':
        normalized = (expression_data.T - expression_data.mean(axis=1)) / expression_data.std(axis=1)
        normalized = normalized.T
    elif method == 'log_zscore':
        log_data = np.log(expression_data + 1)
        normalized = (log_data.T - log_data.mean(axis=1)) / (log_data.std(axis=1) + 1e-8)
        normalized = normalized.T
    elif method == 'minmax':
        min_val = expression_data.min(axis=1)
        max_val = expression_data.max(axis=1)
        normalized = (expression_data.T - min_val) / (max_val - min_val + 1e-8)
        normalized = normalized.T
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # handle NaN and Inf values
    normalized = normalized.fillna(0)
    normalized = normalized.replace([np.inf, -np.inf], 0)

    print(f"  Mean: {normalized.mean().mean():.2e}")
    print(f"  Std: {normalized.std().mean():.2f}")
    print(f"  Range: [{normalized.min().min():.2f}, {normalized.max().max():.2f}]")

    return normalized


def compute_coexpression_network(expression_data, threshold=0.3, method='pearson'):
    """
    Compute gene co-expression network.

    Args:
        expression_data (pd.DataFrame): Normalized expression matrix
        threshold (float): Minimum correlation to create edge
        method (str): Correlation method

    Returns:
        tuple: (edge_index, edge_weight, correlation_tensor)
    """
    print(f"\nComputing co-expression network using {method} correlation")

    num_genes = expression_data.shape[0]
    print(f"  Computing correlations for {num_genes} genes...")
    if method == 'pearson':
        correlation_matrix = expression_data.T.corr(method='pearson')
    elif method == 'spearman':
        correlation_matrix = expression_data.T.corr('spearman')
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # magnitude over direction
    correlation_matrix = correlation_matrix.abs()
    # Set diagonal to 0
    corr_values = correlation_matrix.values.copy()
    np.fill_diagonal(corr_values, 0)
    correlation_matrix = pd.DataFrame(corr_values, index=correlation_matrix.index, columns=correlation_matrix.columns)

    threshold_matrix = (correlation_matrix > threshold).astype(int)

    # convert to tensor
    edge_indices = np.where(threshold_matrix.values == 1)
    edge_index_np = np.stack(edge_indices, axis=0)
    edge_index = torch.from_numpy(edge_index_np).long()

    # edge weights
    edge_weights = torch.tensor(
        correlation_matrix.values[edge_indices],
        dtype=torch.float
    )

    correlation_tensor = torch.tensor(correlation_matrix.values, dtype=torch.float)

    # print graph statistics
    num_edges = edge_index.shape[1]
    density = num_edges / (num_genes * (num_genes - 1))
    print(f"  Nodes: {num_genes}")
    print(f"  Edges: {num_edges}")
    print(f"  Density: {density:.4f}")
    print(f"  Avg degree: {num_edges / num_genes:.2f}")

    return edge_index, edge_weights, correlation_tensor


def load_ground_truth_network(filepath, gene_names, tf_list=None):
    """
    Load ground truth regulatory network.

    Args:
        filepath (str): Path to network CSV file
        gene_names (list): List of gene names in expression data
        tf_list (list): List of TF gene names (optional)

    Returns:
        tuple: (positive_edges, network_df)
    """
    print(f"\nLoading ground truth network from {filepath}")

    network_df = pd.read_csv(filepath)

    # filter to genes present in expression data
    network_df = network_df[
        network_df['Gene1'].isin(gene_names) &
        network_df['Gene2'].isin(gene_names)
    ]

    # if TF list is provided, filter to only TF -> gene edges
    if tf_list is not None:
        network_df = network_df[network_df['Gene1'].isin(tf_list)]

    network_df = network_df[network_df['Gene1'] != network_df['Gene2']]
    network_df = network_df.drop_duplicates(subset=['Gene1', 'Gene2'])

    # create mapping: gene_name -> index
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    source_indices = network_df['Gene1'].map(gene_to_idx)
    target_indices = network_df['Gene2'].map(gene_to_idx)

    edge_array = np.stack([source_indices.values, target_indices.values], axis=0)
    positive_edges = torch.from_numpy(edge_array).long()

    # print stats
    print(f"  Edges: {positive_edges.shape[1]}")
    if tf_list is not None:
        num_tfs = network_df['Gene1'].nunique()
        print(f"  TFs: {num_tfs}")
        print(f"  Target genes: {network_df['Gene2'].nunique()}")

    return positive_edges, network_df


def create_node_features(original_expression_data, normalized_expression_data,
                        gene_names, tf_list=None, cv_threshold=None):
    """
    Create node features for each gene using ORIGINAL expression statistics.

    Features created:
    1. Mean expression (original scale) - baseline expression level
    2. Standard deviation (original scale) - absolute variability
    3. Coefficient of variation (CV%) - relative variability
    4. Is TF indicator - can this gene regulate others?

    Args:
        original_expression_data (pd.DataFrame): Raw expression [genes x samples]
        normalized_expression_data (pd.DataFrame): Normalized [genes x samples]
        gene_names (list): List of gene names to include
        tf_list (list): List of TF gene names
        cv_threshold (float): Optional - filter genes below this CV%

    Returns:
        tuple: (node_features, gene_names_filtered, cv_values)
    """
    print("\nCreating node features...")

    # Filter to only genes we're using
    original_data_filtered = original_expression_data.loc[gene_names]
    num_genes_original = len(gene_names)

    # Calculate statistics on ORIGINAL data
    original_mean = original_data_filtered.mean(axis=1).values
    original_std = original_data_filtered.std(axis=1).values

    # Calculate CV: (std / mean) * 100
    epsilon = 1e-8
    cv = (original_std / (original_mean + epsilon)) * 100

    print(f"  Mean range: [{original_mean.min():.3f}, {original_mean.max():.3f}]")
    print(f"  CV range:   [{cv.min():.2f}%, {cv.max():.2f}%]")

    # Optional: Filter by CV
    if cv_threshold is not None:
        print(f"  Filtering genes with CV < {cv_threshold}%...")
        cv_mask = cv >= cv_threshold

        original_mean = original_mean[cv_mask]
        original_std = original_std[cv_mask]
        cv = cv[cv_mask]
        gene_names = [gene_names[i] for i in range(len(gene_names)) if cv_mask[i]]

        print(f"    Genes before: {num_genes_original}")
        print(f"    Genes after:  {len(gene_names)}")

    num_genes = len(gene_names)

    # Create TF indicator
    if tf_list is not None:
        is_tf = np.array([1.0 if gene in tf_list else 0.0 for gene in gene_names])
        num_tfs = int(is_tf.sum())
        print(f"  TFs: {num_tfs}/{num_genes} ({num_tfs/num_genes*100:.1f}%)")
    else:
        is_tf = np.zeros(num_genes)
        print(f"  No TF list provided")

    # Stack features
    node_features_np = np.stack([
        original_mean,   # Feature 0: baseline expression level
        original_std,    # Feature 1: absolute variability
        cv,              # Feature 2: relative variability
        is_tf            # Feature 3: can regulate others
    ], axis=1)

    node_features = torch.from_numpy(node_features_np).float()

    # Validation
    if torch.isnan(node_features).any():
        num_nan = torch.isnan(node_features).sum().item()
        print(f"  WARNING: {num_nan} NaN values detected!")
    if torch.isinf(node_features).any():
        num_inf = torch.isinf(node_features).sum().item()
        print(f"  WARNING: {num_inf} Inf values detected!")

    print(f"  Node features shape: {node_features.shape}")

    return node_features, gene_names, cv


def create_pytorch_geometric_data(node_features, edge_index, edge_weight,
                                   positive_edges, gene_names, correlation_matrix):
    """
    Create PyTorch Geometric Data object.

    Args:
        node_features (torch.Tensor): Node feature matrix
        edge_index (torch.Tensor): Edge connectivity
        edge_weight (torch.Tensor): Edge weights
        positive_edges (torch.Tensor): Ground truth edges
        gene_names (list): List of gene names
        correlation_matrix (torch.Tensor): Full correlation matrix

    Returns:
        Data: PyTorch Geometric Data object
    """
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1) if edge_weight.numel() > 0 else None,
        y=positive_edges
    )

    data.gene_names = gene_names
    data.correlation_matrix = correlation_matrix

    print(f"  Data object: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.y.shape[1]} positive edges")

    return data


def save_processed_data(data, config, original_expression_data, 
                        normalized_expression_data, cell_type):
    """
    Save processed data to disk in a per-dataset subdirectory.

    Saves to: data/processed/{cell_type}/

    Args:
        data (Data): PyTorch Geometric Data object
        config (dict): Configuration dictionary
        original_expression_data (pd.DataFrame): Original expression data
        normalized_expression_data (pd.DataFrame): Normalized expression data
        cell_type (str): Cell type name (e.g., 'hESC', 'mDC')
    """
    output_dir = os.path.join(config['data']['processed_data_dir'], cell_type)
    os.makedirs(output_dir, exist_ok=True)

    # save Data object
    data_path = os.path.join(output_dir, 'graph_data.pt')
    torch.save(data, data_path)

    # save gene names
    gene_names_path = os.path.join(output_dir, 'gene_names.pkl')
    with open(gene_names_path, 'wb') as f:
        pickle.dump(data.gene_names, f)

    # save original expression data
    original_expr_path = os.path.join(output_dir, 'original_expression.csv')
    original_expression_data.to_csv(original_expr_path)

    # save normalized expression data
    norm_expr_path = os.path.join(output_dir, 'normalized_expression.csv')
    normalized_expression_data.to_csv(norm_expr_path)

    # save processing metadata
    metadata = {
        'cell_type': cell_type,
        'num_genes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_positive_edges': data.y.shape[1] if hasattr(data, 'y') else 0,
        'num_features': data.num_node_features,
        'correlation_threshold': config['data']['correlation_threshold'],
        'max_genes': config['data']['max_genes'],
        'normalization_method': 'log_zscore',
        'has_tf_list': len([f for f in data.x[:, 3] if f == 1]) > 0
    }
    metadata_path = os.path.join(output_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    print(f"  [{cell_type}] Saved to {output_dir}")


def preprocess_single_dataset(cell_type, config):
    """
    Preprocess a single cell type dataset.

    Runs the full pipeline for one cell type:
    1. Load expression data
    2. Normalize
    3. Build co-expression graph
    4. Load ground truth network
    5. Create node features
    6. Package into PyTorch Geometric Data
    7. Save to data/processed/{cell_type}/

    Args:
        cell_type (str): Cell type name (e.g., 'hESC', 'mDC', 'mHSC-E')
        config (dict): Configuration dictionary

    Returns:
        tuple: (Data object, gene_names list)
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {cell_type}")
    print(f"{'='*70}")

    base_dir = config['data']['raw_data_dir']

    # Build paths
    dataset_dir = os.path.join(
        base_dir, 'BEELINE-data', 'inputs', 'scRNA-Seq', cell_type
    )

    expression_file = os.path.join(dataset_dir, 'ExpressionData.csv')
    tf_file = os.path.join(dataset_dir, 'TFs.csv')
    network_file = os.path.join(dataset_dir, 'refNetwork.csv')

    # Check files exist
    if not os.path.exists(expression_file):
        raise FileNotFoundError(f"Expression file not found: {expression_file}")
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network file not found: {network_file}")
    if not os.path.exists(tf_file):
        raise FileNotFoundError(f"TF file not found: {tf_file}")

    # Step 1: Load TF list FIRST (so we can force-include TFs in gene set)
    if os.path.exists(tf_file):
        tf_df = pd.read_csv(tf_file)
        if 'TF' in tf_df.columns:
            tf_list = tf_df['TF'].tolist()
        else:
            tf_list = pd.read_csv(tf_file, header=None)[0].tolist()
        print(f"  Loaded {len(tf_list)} TFs")
    else:
        tf_list = None
        print(f"  No TF file found")

    # Step 2: Load expression data (with TF force-inclusion)
    expression_data = load_expression_data(
        expression_file,
        max_genes=config['data']['max_genes'],
        tf_list=tf_list
    )

    # Save original data BEFORE normalization
    original_expression_data = expression_data.copy()

    # Step 3: Normalize expression
    normalized_data = normalize_expression(expression_data)

    # Step 4: Compute co-expression network
    edge_index, edge_weight, correlation_tensor = compute_coexpression_network(
        normalized_data,
        threshold=config['data']['correlation_threshold']
    )

    # Step 5: Load ground truth network
    positive_edges, network_df = load_ground_truth_network(
        network_file,
        normalized_data.index.tolist(),
        tf_list
    )

    # Step 6: Create node features
    node_features, gene_names_filtered, cv_values = create_node_features(
        original_expression_data=original_expression_data,
        normalized_expression_data=normalized_data,
        gene_names=normalized_data.index.tolist(),
        tf_list=tf_list,
        cv_threshold=None
    )

    # Step 7: Create PyTorch Geometric Data
    data = create_pytorch_geometric_data(
        node_features,
        edge_index,
        edge_weight,
        positive_edges,
        gene_names_filtered,
        correlation_tensor
    )

    # Step 8: Save
    save_processed_data(
        data,
        config,
        original_expression_data.loc[gene_names_filtered],
        normalized_data.loc[gene_names_filtered],
        cell_type
    )

    print(f"  [{cell_type}] Done: {data.num_nodes} genes, "
          f"{data.num_edges} co-expression edges, "
          f"{data.y.shape[1]} ground truth edges")

    return data, gene_names_filtered


def preprocess_pipeline(config):
    """
    Main preprocessing pipeline. Processes all datasets defined in config.

    Reads config['data']['datasets']['train'] and config['data']['datasets']['test'],
    then calls preprocess_single_dataset() for each cell type.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: {cell_type: (Data, gene_names)} for all processed datasets
    """
    print("\n" + "="*70)
    print("GeneInference Multi-Dataset Preprocessing Pipeline")
    print("="*70)

    all_dataset_names = (
        config['data']['datasets']['train'] +
        config['data']['datasets']['test']
    )

    print(f"\nDatasets to process: {len(all_dataset_names)}")
    print(f"  Train: {config['data']['datasets']['train']}")
    print(f"  Test:  {config['data']['datasets']['test']}")

    results = {}
    for dataset_name in all_dataset_names:
        cell_type = dataset_name.replace('beeline_', '')
        data, gene_names = preprocess_single_dataset(cell_type, config)
        results[cell_type] = (data, gene_names)

    # Summary
    print(f"\n{'='*70}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Cell Type':<12} {'Genes':>6} {'Co-expr Edges':>14} {'GT Edges':>10} {'TFs':>5}")
    print("-" * 52)
    for cell_type, (data, gene_names) in results.items():
        num_tfs = int(data.x[:, 3].sum().item())
        print(f"{cell_type:<12} {data.num_nodes:>6} {data.num_edges:>14} "
              f"{data.y.shape[1]:>10} {num_tfs:>5}")
    print(f"{'='*70}\n")

    return results


def save_metadata(config, output_dir):
    """Save preprocessing metadata"""
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, 'preprocessing_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(config, f)


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess data for GeneInference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = preprocess_pipeline(config)

    print("\nNext step: python main.py --experiment-type train\n")


if __name__ == '__main__':
    main()
