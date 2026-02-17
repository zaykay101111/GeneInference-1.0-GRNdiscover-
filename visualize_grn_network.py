"""
Visualize GRN Network Graph
============================

This script creates a network visualization of predicted gene regulatory networks
from predicted_grn.csv files, similar to Cytoscape-style layouts.

Author: Kyler Zook
Date: January 2026
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import numpy as np


def load_predictions(csv_path, top_k=None, min_probability=0.5):
    """
    Load GRN predictions from CSV.

    Args:
        csv_path: Path to predicted_grn.csv
        top_k: Only keep top K predictions (by probability)
        min_probability: Minimum prediction probability to include

    Returns:
        DataFrame with filtered predictions
    """
    df = pd.read_csv(csv_path)

    # Filter by minimum probability
    df = df[df['probability'] >= min_probability].copy()

    # Sort by probability descending
    df = df.sort_values('probability', ascending=False)

    # Keep only top K if specified
    if top_k is not None:
        df = df.head(top_k)

    print(f"Loaded {len(df)} edges (filtered from CSV)")
    print(f"  TFs: {df['source_gene'].nunique()}")
    print(f"  Target genes: {df['target_gene'].nunique()}")
    print(f"  Probability range: [{df['probability'].min():.3f}, {df['probability'].max():.3f}]")

    return df


def create_grn_graph(predictions_df):
    """
    Create NetworkX directed graph from predictions.

    Args:
        predictions_df: DataFrame with source_gene, target_gene, probability

    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()

    # Add edges with weights
    for _, row in predictions_df.iterrows():
        G.add_edge(
            row['source_gene'],
            row['target_gene'],
            weight=row['probability']
        )

    return G


def visualize_grn_network(G, predictions_df, output_path, layout='spring', figsize=(20, 20)):
    """
    Visualize GRN network graph.

    Args:
        G: NetworkX graph
        predictions_df: Original predictions DataFrame
        output_path: Path to save figure
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        figsize: Figure size (width, height)
    """
    print(f"\nCreating network visualization with {layout} layout...")

    # Get all unique genes
    all_genes = set(predictions_df['source_gene'].unique()) | set(predictions_df['target_gene'].unique())

    # Identify TFs (source genes) and target genes
    tfs = set(predictions_df['source_gene'].unique())
    targets = set(predictions_df['target_gene'].unique())

    # Genes that are both TF and target
    both = tfs & targets
    # TFs only (not targets)
    tf_only = tfs - targets
    # Target genes only (not TFs)
    target_only = targets - tfs

    print(f"  TFs only: {len(tf_only)}")
    print(f"  Target genes only: {len(target_only)}")
    print(f"  Both TF and target: {len(both)}")

    # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges with varying thickness and transparency based on prediction probability
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Normalize weights for edge width (0.5 to 3.0)
    min_weight = min(weights)
    max_weight = max(weights)
    if max_weight > min_weight:
        edge_widths = [0.5 + 2.5 * (w - min_weight) / (max_weight - min_weight) for w in weights]
    else:
        edge_widths = [1.5] * len(weights)

    # Edge colors based on weight (lighter = lower probability)
    edge_colors = [(0.3, 0.3, 0.3, 0.2 + 0.6 * (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5)
                   for w in weights]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=10,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        ax=ax,
        alpha=0.6
    )

    # Draw nodes with different colors for TFs vs targets
    # TF-only nodes (orange)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(tf_only),
        node_color='#FF8C00',  # Dark orange
        node_size=800,
        node_shape='o',
        edgecolors='black',
        linewidths=2,
        ax=ax,
        label='Transcription Factor'
    )

    # Target-only nodes (light gray)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(target_only),
        node_color='#D3D3D3',  # Light gray
        node_size=600,
        node_shape='o',
        edgecolors='gray',
        linewidths=1.5,
        ax=ax,
        label='Target Gene'
    )

    # Both TF and target (yellow/gold)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(both),
        node_color='#FFD700',  # Gold
        node_size=800,
        node_shape='o',
        edgecolors='black',
        linewidths=2,
        ax=ax,
        label='TF & Target'
    )

    # Draw labels
    # Only show labels for TFs (to avoid clutter)
    tf_labels = {node: node for node in G.nodes() if node in tfs}

    nx.draw_networkx_labels(
        G, pos,
        labels=tf_labels,
        font_size=8,
        font_weight='bold',
        font_color='black',
        ax=ax
    )

    # Title and legend
    ax.set_title(
        f'Gene Regulatory Network\n{len(G.nodes())} genes, {len(G.edges())} predicted regulatory edges',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.legend(
        loc='upper left',
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    ax.axis('off')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved network visualization to {output_path}")

    plt.close()


def visualize_grn_hierarchical(G, predictions_df, output_path, figsize=(24, 16)):
    """
    Create hierarchical layout with TFs on the left and targets on the right.

    Args:
        G: NetworkX graph
        predictions_df: Original predictions DataFrame
        output_path: Path to save figure
        figsize: Figure size
    """
    print(f"\nCreating hierarchical network visualization...")

    # Identify TFs and targets
    tfs = set(predictions_df['source_gene'].unique())
    targets = set(predictions_df['target_gene'].unique())
    both = tfs & targets
    tf_only = tfs - targets
    target_only = targets - tfs

    # Create hierarchical positions
    pos = {}

    # TFs on the left (x=0)
    tf_list = sorted(list(tf_only | both))
    for i, tf in enumerate(tf_list):
        pos[tf] = (0, i / max(len(tf_list) - 1, 1))

    # Targets on the right (x=1)
    target_list = sorted(list(target_only | both))
    for i, target in enumerate(target_list):
        if target not in pos:  # Only add if not already positioned as TF
            pos[target] = (1, i / max(len(target_list) - 1, 1))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    min_weight = min(weights)
    max_weight = max(weights)
    if max_weight > min_weight:
        edge_widths = [0.3 + 1.5 * (w - min_weight) / (max_weight - min_weight) for w in weights]
        edge_alphas = [0.1 + 0.4 * (w - min_weight) / (max_weight - min_weight) for w in weights]
    else:
        edge_widths = [1.0] * len(weights)
        edge_alphas = [0.3] * len(weights)

    # Draw each edge individually with its alpha
    for (u, v), width, alpha in zip(edges, edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=[(0.3, 0.3, 0.3, alpha)],
            arrows=True,
            arrowsize=8,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.05',
            ax=ax
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(tf_only),
        node_color='#FF8C00',
        node_size=600,
        edgecolors='black',
        linewidths=2,
        ax=ax,
        label='Transcription Factor'
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(target_only),
        node_color='#D3D3D3',
        node_size=400,
        edgecolors='gray',
        linewidths=1,
        ax=ax,
        label='Target Gene'
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(both),
        node_color='#FFD700',
        node_size=600,
        edgecolors='black',
        linewidths=2,
        ax=ax,
        label='TF & Target'
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=7,
        font_weight='bold',
        ax=ax
    )

    ax.set_title(
        f'Gene Regulatory Network (Hierarchical Layout)\n{len(G.nodes())} genes, {len(G.edges())} predicted edges',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved hierarchical visualization to {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predicted GRN as network graph'
    )

    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predicted_grn.csv file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (defaults to same dir as predictions)')
    parser.add_argument('--top-k', type=int, default=200,
                       help='Only visualize top K predictions (default: 200)')
    parser.add_argument('--min-prob', type=float, default=0.7,
                       help='Minimum prediction probability (default: 0.7)')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'circular', 'kamada_kawai', 'spectral', 'hierarchical', 'all'],
                       help='Layout algorithm')
    parser.add_argument('--figsize', type=int, nargs=2, default=[20, 20],
                       help='Figure size in inches (width height)')

    args = parser.parse_args()

    print("=" * 80)
    print("GENE REGULATORY NETWORK VISUALIZATION")
    print("=" * 80)

    # Load predictions
    predictions_df = load_predictions(
        args.predictions,
        top_k=args.top_k,
        min_probability=args.min_prob
    )

    if len(predictions_df) == 0:
        print("\n✗ No predictions match the filtering criteria")
        print(f"  Try lowering --min-prob (current: {args.min_prob})")
        return

    # Create graph
    G = create_grn_graph(predictions_df)

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create visualizations
    if args.layout == 'all':
        layouts = ['spring', 'circular', 'kamada_kawai', 'hierarchical']
    elif args.layout == 'hierarchical':
        layouts = ['hierarchical']
    else:
        layouts = [args.layout]

    for layout in layouts:
        if layout == 'hierarchical':
            output_path = os.path.join(args.output_dir, 'grn_network_hierarchical.png')
            visualize_grn_hierarchical(
                G, predictions_df, output_path,
                figsize=tuple(args.figsize)
            )
        else:
            output_path = os.path.join(args.output_dir, f'grn_network_{layout}.png')
            visualize_grn_network(
                G, predictions_df, output_path,
                layout=layout,
                figsize=tuple(args.figsize)
            )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
