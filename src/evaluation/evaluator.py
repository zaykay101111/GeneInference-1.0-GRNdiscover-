import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import yaml
import argparse
import pickle
import sys
from pathlib import Path

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.architectures import create_model
from src.utils.helpers import load_checkpoint, compute_metrics
from src.data.dataset import create_dataloaders, load_single_dataset, create_test_dataloader


def load_config(config_path='config.yaml'):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_roc_curve(labels, predictions, save_path):
    """
    Plot ROC curve.

    ROC (Receiver Operating Characteristic) curve shows tradeoff between
    true positive rate and false positive rate at different thresholds.

    Args:
        labels: True binary labels
        predictions: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})', color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random classifier (AUC = 0.50)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - GRN Edge Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved ROC curve to {save_path}")
    print(f"  AUROC: {roc_auc:.4f}")


def plot_pr_curve(labels, predictions, save_path):
    """
    Plot Precision-Recall curve.

    PR curve is often more informative than ROC for imbalanced datasets
    (which GRN networks are - very few true edges compared to possible edges).

    Args:
        labels: True binary labels
        predictions: Predicted probabilities
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    # Baseline (random classifier) for imbalanced data
    baseline = labels.sum() / len(labels)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {avg_precision:.4f})', color='#A23B72')
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5,
                label=f'Random classifier (AP = {baseline:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - GRN Edge Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved PR curve to {save_path}")
    print(f"  AUPRC: {avg_precision:.4f}")


def plot_confusion_matrix(labels, predictions, threshold, save_path):
    """
    Plot confusion matrix.

    Shows true positives, false positives, true negatives, false negatives.

    Args:
        labels: True binary labels
        predictions: Predicted probabilities
        threshold: Classification threshold
        save_path: Path to save figure
    """
    # Convert probabilities to binary predictions
    pred_binary = (predictions >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(labels, pred_binary)

    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: Non-regulatory', 'Predicted: Regulatory'],
                yticklabels=['Actual: Non-regulatory', 'Actual: Regulatory'],
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix (threshold = {threshold:.2f})',
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add metrics as text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    stats_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    plt.text(2.5, 0.5, stats_text, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved confusion matrix to {save_path}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")


def plot_threshold_analysis(labels, predictions, save_path):
    """
    Plot precision, recall, and F1 score across different thresholds.

    Helps choose optimal threshold for different use cases.

    Args:
        labels: True binary labels
        predictions: Predicted probabilities
        save_path: Path to save figure
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        pred_binary = (predictions >= threshold).astype(int)

        tp = ((pred_binary == 1) & (labels == 1)).sum()
        fp = ((pred_binary == 1) & (labels == 0)).sum()
        fn = ((pred_binary == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, linewidth=2, label='Precision', color='#2E86AB')
    plt.plot(thresholds, recalls, linewidth=2, label='Recall', color='#A23B72')
    plt.plot(thresholds, f1_scores, linewidth=2, label='F1 Score', color='#F18F01')

    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.axvline(x=optimal_threshold, color='green', linestyle='--',
                linewidth=1.5, label=f'Optimal F1 threshold = {optimal_threshold:.2f}')

    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved threshold analysis to {save_path}")
    print(f"  Optimal F1 threshold: {optimal_threshold:.2f}")


def plot_prediction_heatmap(labels, predictions, threshold, save_path):
    """
    Plot confusion heatmap showing relationship between predicted and actual edges.

    Creates a 2x2 heatmap showing:
    - True Positives (Predicted Regulatory & Actually Regulatory)
    - False Positives (Predicted Regulatory & Actually Non-Regulatory)
    - False Negatives (Predicted Non-Regulatory & Actually Regulatory)
    - True Negatives (Predicted Non-Regulatory & Actually Non-Regulatory)

    Args:
        labels: True binary labels (1=regulatory, 0=non-regulatory)
        predictions: Predicted probabilities [0, 1]
        threshold: Classification threshold
        save_path: Path to save figure
    """
    # Convert probabilities to binary predictions
    predicted_labels = (predictions >= threshold).astype(int)

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predicted_labels)

    # Calculate percentages
    cm_percentages = cm.astype('float') / cm.sum() * 100

    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap - use a diverging colormap centered on middle values
    import matplotlib.colors as mcolors
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue', colors, N=n_bins)

    # Create heatmap with annotations
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Edges', rotation=270, labelpad=20, fontsize=12)

    # Set ticks and labels
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Non-Regulatory\n(Predicted)', 'Regulatory\n(Predicted)'], fontsize=12)
    ax.set_yticklabels(['Non-Regulatory\n(Actual)', 'Regulatory\n(Actual)'], fontsize=12)

    # Add title and labels
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Class', fontsize=14, fontweight='bold')
    plt.title('Prediction vs Ground Truth Heatmap', fontsize=16, fontweight='bold', pad=20)

    # Add text annotations with counts and percentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = cm_percentages[i, j]

            # Determine cell type
            if i == 1 and j == 1:
                cell_type = 'True Positive'
                color = 'white' if count > cm.max() / 2 else 'black'
            elif i == 0 and j == 1:
                cell_type = 'False Positive'
                color = 'white' if count > cm.max() / 2 else 'black'
            elif i == 1 and j == 0:
                cell_type = 'False Negative'
                color = 'white' if count > cm.max() / 2 else 'black'
            else:
                cell_type = 'True Negative'
                color = 'white' if count > cm.max() / 2 else 'black'

            # Add text with count and percentage
            text = ax.text(j, i, f'{cell_type}\n\n{count:,}\n({percentage:.1f}%)',
                          ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    # Add grid
    ax.set_xticks(np.arange(2) - 0.5, minor=True)
    ax.set_yticks(np.arange(2) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    print(f"✓ Saved prediction heatmap to {save_path}")
    print(f"\n  Confusion Matrix Summary:")
    print(f"    True Positives:  {tp:6,} ({tp/total*100:5.1f}%) - Correctly predicted regulatory edges")
    print(f"    False Positives: {fp:6,} ({fp/total*100:5.1f}%) - Incorrectly predicted as regulatory")
    print(f"    False Negatives: {fn:6,} ({fn/total*100:5.1f}%) - Missed regulatory edges")
    print(f"    True Negatives:  {tn:6,} ({tn/total*100:5.1f}%) - Correctly predicted non-regulatory")
    print(f"    Total:           {total:6,}")


def analyze_per_tf_performance(edges_to_predict, predictions, labels, gene_names, save_path):
    """
    Analyze performance per transcription factor.

    Some TFs may be easier to predict than others.
    This analysis helps identify which TFs the model struggles with.

    Args:
        edges_to_predict: Edge indices [2, num_edges]
        predictions: Predicted probabilities
        labels: True labels
        gene_names: List of gene names
        save_path: Path to save figure
    """
    # Group by source TF (source node)
    source_nodes = edges_to_predict[0].numpy()

    # Create DataFrame for analysis
    df = pd.DataFrame({
        'source_tf': source_nodes,
        'prediction': predictions.numpy(),
        'label': labels.numpy()
    })

    # Compute metrics per TF
    tf_metrics = []
    for tf_idx in df['source_tf'].unique():
        tf_data = df[df['source_tf'] == tf_idx]

        if len(tf_data) < 5:  # Skip TFs with too few edges
            continue

        # Compute AUPRC for this TF
        if len(tf_data['label'].unique()) == 2:  # Need both classes
            auprc = average_precision_score(tf_data['label'], tf_data['prediction'])
        else:
            auprc = 0.0

        tf_metrics.append({
            'tf_idx': tf_idx,
            'tf_name': gene_names[tf_idx] if tf_idx < len(gene_names) else f'Gene_{tf_idx}',
            'num_edges': len(tf_data),
            'num_positive': tf_data['label'].sum(),
            'auprc': auprc
        })

    tf_metrics_df = pd.DataFrame(tf_metrics)

    # Check if we have any TF metrics to analyze
    if len(tf_metrics_df) == 0:
        print("  ⚠ Not enough TFs with sufficient edges for per-TF analysis (need at least 5 edges per TF)")
        print("  ⚠ Skipping per-TF performance plot")
        return

    tf_metrics_df = tf_metrics_df.sort_values('auprc', ascending=False)

    # Save to CSV
    csv_path = save_path.replace('.png', '.csv')
    tf_metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved per-TF metrics to {csv_path}")

    # Plot top and bottom performing TFs
    n_show = min(10, len(tf_metrics_df))
    top_tfs = tf_metrics_df.head(n_show)
    bottom_tfs = tf_metrics_df.tail(n_show)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Top performing TFs
    ax1.barh(range(len(top_tfs)), top_tfs['auprc'].values, color='#2E86AB')
    ax1.set_yticks(range(len(top_tfs)))
    ax1.set_yticklabels(top_tfs['tf_name'].values, fontsize=9)
    ax1.set_xlabel('AUPRC', fontsize=11)
    ax1.set_title(f'Top {n_show} Performing TFs', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Bottom performing TFs
    ax2.barh(range(len(bottom_tfs)), bottom_tfs['auprc'].values, color='#A23B72')
    ax2.set_yticks(range(len(bottom_tfs)))
    ax2.set_yticklabels(bottom_tfs['tf_name'].values, fontsize=9)
    ax2.set_xlabel('AUPRC', fontsize=11)
    ax2.set_title(f'Bottom {n_show} Performing TFs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved per-TF analysis to {save_path}")
    print(f"  Best TF: {top_tfs.iloc[0]['tf_name']} (AUPRC: {top_tfs.iloc[0]['auprc']:.3f})")
    print(f"  Worst TF: {bottom_tfs.iloc[-1]['tf_name']} (AUPRC: {bottom_tfs.iloc[-1]['auprc']:.3f})")


def generate_evaluation_report(results, save_path):
    """
    Generate a comprehensive evaluation report.

    Args:
        results: Dictionary with all evaluation results
        save_path: Path to save report
    """
    report = []
    report.append("=" * 70)
    report.append("SimpleGRN Evaluation Report")
    report.append("=" * 70)
    report.append("")

    report.append("Model Configuration:")
    report.append(f"  Model type: {results['config']['model']['model_type']}")
    report.append(f"  Hidden dim: {results['config']['model']['hidden_dim']}")
    report.append(f"  Num layers: {results['config']['model']['num_layers']}")
    report.append(f"  Dropout: {results['config']['model']['dropout']}")
    report.append("")

    report.append("Dataset Statistics:")
    report.append(f"  Total genes: {results['num_nodes']}")
    report.append(f"  Test edges: {results['num_test_edges']}")
    report.append(f"  Positive edges: {results['num_positive']}")
    report.append(f"  Negative edges: {results['num_negative']}")
    report.append(f"  Class balance: {results['num_positive']/results['num_test_edges']*100:.2f}% positive")
    report.append("")

    report.append("Performance Metrics:")
    report.append(f"  AUROC:     {results['metrics']['auroc']:.4f}")
    report.append(f"  AUPRC:     {results['metrics']['auprc']:.4f}")
    report.append(f"  F1 Score:  {results['metrics']['f1']:.4f}")
    report.append(f"  Precision: {results['metrics']['precision']:.4f}")
    report.append(f"  Recall:    {results['metrics']['recall']:.4f}")
    report.append("")

    report.append("Interpretation:")
    auroc = results['metrics']['auroc']
    if auroc >= 0.95:
        report.append("  ✓ EXCELLENT - Very high discrimination (>0.95)")
    elif auroc >= 0.90:
        report.append("  ✓ VERY GOOD - High discrimination (0.90-0.95)")
    elif auroc >= 0.80:
        report.append("  ✓ GOOD - Strong performance (0.80-0.90)")
    elif auroc >= 0.70:
        report.append("  ✓ FAIR - Acceptable performance (0.70-0.80)")
    else:
        report.append("  ⚠ NEEDS IMPROVEMENT - Below 0.70")

    report.append("")
    report.append(f"  Recall = {results['metrics']['recall']:.2f}:")
    if results['metrics']['recall'] >= 0.99:
        report.append("    → Model catches ALL positive cases (no false negatives)")
    elif results['metrics']['recall'] >= 0.90:
        report.append("    → Model catches most positive cases (few false negatives)")
    else:
        report.append("    → Model misses some positive cases")

    report.append("")
    report.append(f"  Precision = {results['metrics']['precision']:.4f}:")
    report.append(f"    → {results['metrics']['precision']*100:.1f}% of predictions are correct")

    report.append("")
    report.append("=" * 70)

    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))

    # Print report
    print('\n'.join(report))


def parse_model_config_from_path(checkpoint_path):
    """
    Parse model configuration from checkpoint path.

    Expected format: ./experiments/{model}_{activation}_h{hidden}_l{layers}/checkpoints/best_model.pt

    Returns:
        dict: Model configuration with model_type, activation, hidden_dim, num_layers
    """
    import re

    # Extract experiment name from path
    # e.g., ./experiments/gin_relu_h64_l2/checkpoints/best_model.pt -> gin_relu_h64_l2
    path_parts = checkpoint_path.replace('\\', '/').split('/')

    # Find the experiment directory (parent of 'checkpoints')
    exp_name = None
    for i, part in enumerate(path_parts):
        if part == 'checkpoints' and i > 0:
            exp_name = path_parts[i - 1]
            break

    if not exp_name:
        return None

    # Parse experiment name: {model}_{activation}_h{hidden}_l{layers}
    parts = exp_name.split('_')

    if len(parts) < 4:
        return None

    # Extract components
    model_type = parts[0]

    # Hidden dim (e.g., h64)
    hidden_dim = 64
    for part in parts:
        if part.startswith('h') and part[1:].isdigit():
            hidden_dim = int(part[1:])
            break

    # Num layers (e.g., l2)
    num_layers = 2
    for part in parts:
        if part.startswith('l') and part[1:].isdigit():
            num_layers = int(part[1:])
            break

    # Activation is everything between model_type and h{}/l{}
    activation_parts = []
    for part in parts[1:]:
        if part.startswith('h') or part.startswith('l'):
            break
        activation_parts.append(part)
    activation = '_'.join(activation_parts) if activation_parts else 'relu'

    return {
        'model_type': model_type,
        'activation': activation,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate SimpleGRN model')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots (auto-generated if not specified)')
    parser.add_argument('--threshold', type=float, default=None, help='Classification threshold (overrides checkpoint threshold)')
    parser.add_argument('--find-threshold', action='store_true', help='Find optimal threshold on test set')
    args = parser.parse_args()

    # Load base config
    config = load_config(args.config)

    # Parse model config from checkpoint path
    model_config = parse_model_config_from_path(args.checkpoint)
    if model_config:
        print(f"\nDetected model configuration from checkpoint path:")
        print(f"  Model: {model_config['model_type']}")
        print(f"  Activation: {model_config['activation']}")
        print(f"  Hidden dim: {model_config['hidden_dim']}")
        print(f"  Layers: {model_config['num_layers']}")

        # Override config with detected values
        config['model']['model_type'] = model_config['model_type']
        config['model']['activation'] = model_config['activation']
        config['model']['hidden_dim'] = model_config['hidden_dim']
        config['model']['num_layers'] = model_config['num_layers']
    else:
        print("\nWARNING: Could not parse model config from checkpoint path.")
        print("Using config from config.yaml...")

    print("=" * 70)
    print("SimpleGRN Evaluation")
    print("=" * 70)

    # Auto-generate output directory based on checkpoint path if not specified
    if args.output_dir is None:
        # Extract experiment name from checkpoint path
        # e.g., ./experiments/mlp_leaky_relu_h64_l2/checkpoints/best_model.pt -> mlp_leaky_relu_h64_l2
        checkpoint_path = os.path.normpath(args.checkpoint)
        path_parts = checkpoint_path.split(os.sep)

        # Try to find experiment name (parent of 'checkpoints' folder)
        if 'checkpoints' in path_parts:
            checkpoints_idx = path_parts.index('checkpoints')
            if checkpoints_idx > 0:
                experiment_name = path_parts[checkpoints_idx - 1]
            else:
                experiment_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        else:
            # Use checkpoint filename without extension
            experiment_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

        args.output_dir = os.path.join('results', f'eval_{experiment_name}')
        print(f"\nAuto-generated output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    print(f"\nUsing device: {device}")

    # ===================================================================
    # 1. Load Data
    # ===================================================================
    print("\nLoading data...")
    data_path = os.path.join(config['data']['processed_data_dir'], 'graph_data.pt')
    data = torch.load(data_path, weights_only=False)

    # Load gene names
    gene_names_path = os.path.join(config['data']['processed_data_dir'], 'gene_names.pkl')
    with open(gene_names_path, 'rb') as f:
        gene_names = pickle.load(f)

    print(f"  Nodes: {data.num_nodes}")
    print(f"  Features: {data.num_node_features}")

    # Try to load saved edge splits from checkpoint directory
    # This ensures we use the same test set as during training
    checkpoint_dir = os.path.dirname(args.checkpoint)
    splits_path = os.path.join(checkpoint_dir, 'edge_splits.pkl')

    if os.path.exists(splits_path):
        print(f"\n  Found saved edge splits at {splits_path}")
        _, _, test_loader, _ = create_dataloaders(data, config, splits_path=splits_path)
    else:
        print(f"\n  WARNING: No saved edge splits found at {splits_path}")
        print("  Creating new splits - results may differ from training!")
        _, _, test_loader, _ = create_dataloaders(data, config)

    # ===================================================================
    # 2. Load Model
    # ===================================================================
    print("\nLoading model...")
    model = create_model(config, data.num_node_features)
    checkpoint = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Check if checkpoint has optimal threshold
    if 'optimal_threshold' in checkpoint and checkpoint['optimal_threshold'] is not None:
        optimal_threshold = checkpoint['optimal_threshold']
        print(f"  Using optimal threshold from checkpoint: {optimal_threshold:.4f}")
    else:
        optimal_threshold = config['evaluation'].get('threshold', 0.5)
        print(f"  No optimal threshold in checkpoint, using config default: {optimal_threshold:.4f}")

    # Override with command-line threshold if provided
    if args.threshold is not None:
        optimal_threshold = args.threshold
        print(f"  ⚠ Overriding with command-line threshold: {optimal_threshold:.4f}")

    # ===================================================================
    # 3. Get Predictions
    # ===================================================================
    print("\nGenerating predictions on test set...")

    all_predictions = []
    all_labels = []
    all_edges = []

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        for batch in test_loader:
            edges = batch['edges'].to(device)
            labels = batch['labels'].to(device)
            correlations = batch['correlations'].to(device)

            # Forward pass
            predictions = model(x, edge_index, edges, correlations)
            predictions = torch.sigmoid(predictions)  # Convert logits to probabilities

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_edges.append(edges.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_edges = torch.cat(all_edges, dim=1)

    print(f"  Total test samples: {len(all_labels)}")
    print(f"  Positive samples: {all_labels.sum().item()}")
    print(f"  Negative samples: {(all_labels == 0).sum().item()}")

    # Find optimal threshold if requested
    if args.find_threshold:
        from utils import find_optimal_threshold
        print("\nFinding optimal threshold on test set...")
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            all_predictions, all_labels, metric='f1'
        )
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"    Precision: {threshold_metrics['precision']:.4f}")
        print(f"    Recall: {threshold_metrics['recall']:.4f}")
        print(f"    F1: {threshold_metrics['f1']:.4f}")

    # ===================================================================
    # 4. Compute Metrics
    # ===================================================================
    print("\nComputing metrics...")
    metrics = compute_metrics(all_predictions, all_labels, config['evaluation']['metrics'], threshold=optimal_threshold)

    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # ===================================================================
    # 5. Generate Visualizations
    # ===================================================================
    print("\nGenerating visualizations...")

    # ROC Curve
    plot_roc_curve(
        all_labels.numpy(),
        all_predictions.numpy(),
        os.path.join(args.output_dir, 'roc_curve.png')
    )

    # PR Curve
    plot_pr_curve(
        all_labels.numpy(),
        all_predictions.numpy(),
        os.path.join(args.output_dir, 'pr_curve.png')
    )

    # Confusion Matrix
    plot_confusion_matrix(
        all_labels.numpy(),
        all_predictions.numpy(),
        optimal_threshold,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    # Threshold Analysis
    plot_threshold_analysis(
        all_labels.numpy(),
        all_predictions.numpy(),
        os.path.join(args.output_dir, 'threshold_analysis.png')
    )

    # Prediction Heatmap (2x2 with TP/FP/FN/TN labels)
    plot_prediction_heatmap(
        all_labels.numpy(),
        all_predictions.numpy(),
        optimal_threshold,
        os.path.join(args.output_dir, 'prediction_heatmap.png')
    )

    # Per-TF Analysis
    analyze_per_tf_performance(
        all_edges,
        all_predictions,
        all_labels,
        gene_names,
        os.path.join(args.output_dir, 'per_tf_performance.png')
    )

    # ===================================================================
    # 6. Generate Report
    # ===================================================================
    print("\nGenerating evaluation report...")

    results = {
        'config': config,
        'num_nodes': data.num_nodes,
        'num_test_edges': len(all_labels),
        'num_positive': all_labels.sum().item(),
        'num_negative': (all_labels == 0).sum().item(),
        'metrics': metrics
    }

    generate_evaluation_report(
        results,
        os.path.join(args.output_dir, 'evaluation_report.txt')
    )

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - roc_curve.png")
    print("  - pr_curve.png")
    print("  - confusion_matrix.png")
    print("  - prediction_heatmap.png")
    print("  - threshold_analysis.png")
    print("  - per_tf_performance.png")
    print("  - per_tf_performance.csv")
    print("  - evaluation_report.txt")


def _evaluate_single_dataset(model, data, gene_names, test_loader, optimal_threshold,
                              config, device, output_dir, prefix=''):
    """
    Run evaluation on a single dataset's test loader.

    Args:
        model: Trained model
        data: PyTorch Geometric Data object
        gene_names: List of gene names
        test_loader: DataLoader for test edges
        optimal_threshold: Classification threshold
        config: Configuration dict
        device: Device
        output_dir: Directory to save plots
        prefix: Filename prefix (e.g., 'mESC_')

    Returns:
        dict: Evaluation results
    """
    all_predictions = []
    all_labels = []
    all_edges = []

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        for batch in test_loader:
            edges = batch['edges'].to(device)
            labels = batch['labels'].to(device)
            correlations = batch['correlations'].to(device)

            predictions = model(x, edge_index, edges, correlations)
            predictions = torch.sigmoid(predictions)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_edges.append(edges.cpu())

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_edges = torch.cat(all_edges, dim=1)

    # Compute Metrics
    metrics = compute_metrics(all_predictions, all_labels,
                              config['evaluation']['metrics'], threshold=optimal_threshold)

    # Generate Visualizations
    plot_roc_curve(
        all_labels.numpy(), all_predictions.numpy(),
        os.path.join(output_dir, f'{prefix}roc_curve.png')
    )
    plot_pr_curve(
        all_labels.numpy(), all_predictions.numpy(),
        os.path.join(output_dir, f'{prefix}pr_curve.png')
    )
    plot_confusion_matrix(
        all_labels.numpy(), all_predictions.numpy(), optimal_threshold,
        os.path.join(output_dir, f'{prefix}confusion_matrix.png')
    )
    plot_prediction_heatmap(
        all_labels.numpy(), all_predictions.numpy(), optimal_threshold,
        os.path.join(output_dir, f'{prefix}prediction_heatmap.png')
    )
    plot_threshold_analysis(
        all_labels.numpy(), all_predictions.numpy(),
        os.path.join(output_dir, f'{prefix}threshold_analysis.png')
    )
    analyze_per_tf_performance(
        all_edges, all_predictions, all_labels, gene_names,
        os.path.join(output_dir, f'{prefix}per_tf_performance.png')
    )

    results = {
        'config': config,
        'num_nodes': data.num_nodes,
        'num_test_edges': len(all_labels),
        'num_positive': int(all_labels.sum().item()),
        'num_negative': int((all_labels == 0).sum().item()),
        'metrics': {k: float(v) for k, v in metrics.items()}
    }

    generate_evaluation_report(
        results,
        os.path.join(output_dir, f'{prefix}evaluation_report.txt')
    )

    return results


def run_evaluation(config, checkpoint_path, output_dir):
    """
    Run evaluation programmatically.

    Supports both single-dataset and multi-dataset modes.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for results

    Returns:
        dict: Evaluation results including metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and
                           config['training']['use_gpu'] else 'cpu')

    # Determine if multi-dataset mode
    use_multi = ('datasets' in config['data'] and
                 'test' in config['data']['datasets'] and
                 len(config['data']['datasets']['test']) > 0)

    if use_multi:
        # Load first training dataset to get num_node_features
        first_train = config['data']['datasets']['train'][0].replace('beeline_', '')
        first_data, _ = load_single_dataset(config, first_train)
        num_node_features = first_data.num_node_features

        # Load Model
        model = create_model(config, num_node_features)
        checkpoint = load_checkpoint(checkpoint_path, model)
        model = model.to(device)
        model.eval()

        if 'optimal_threshold' in checkpoint and checkpoint['optimal_threshold'] is not None:
            optimal_threshold = checkpoint['optimal_threshold']
        else:
            optimal_threshold = config['evaluation'].get('threshold', 0.5)

        # Evaluate each test dataset
        all_results = {}
        test_dataset_names = config['data']['datasets']['test']

        for dataset_name in test_dataset_names:
            cell_type = dataset_name.replace('beeline_', '')
            print(f"\n{'='*60}")
            print(f"Evaluating: {cell_type}")
            print(f"{'='*60}")

            data, gene_names = load_single_dataset(config, cell_type)

            # Create test loader using ALL GT edges
            from src.data.dataset import collate_fn, GRNEdgeDataset
            test_dataset = GRNEdgeDataset(
                data=data, split='test',
                split_edges={'test': data.y},
                neg_sampling_ratio=config['loss']['neg_sampling_ratio']
            )
            from torch.utils.data import DataLoader
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False, collate_fn=collate_fn
            )

            ct_output_dir = os.path.join(output_dir, cell_type)
            os.makedirs(ct_output_dir, exist_ok=True)

            ct_results = _evaluate_single_dataset(
                model, data, gene_names, test_loader,
                optimal_threshold, config, device, ct_output_dir
            )
            all_results[cell_type] = ct_results

        # Compute average metrics
        metric_names = list(next(iter(all_results.values()))['metrics'].keys())
        avg_metrics = {}
        for mn in metric_names:
            vals = [r['metrics'][mn] for r in all_results.values()]
            avg_metrics[mn] = sum(vals) / len(vals)

        combined_results = {
            'config': config,
            'per_dataset': {ct: r['metrics'] for ct, r in all_results.items()},
            'metrics': avg_metrics,
            'num_test_edges': sum(r['num_test_edges'] for r in all_results.values()),
            'num_positive': sum(r['num_positive'] for r in all_results.values()),
            'num_negative': sum(r['num_negative'] for r in all_results.values()),
            'num_nodes': sum(r['num_nodes'] for r in all_results.values()),
        }

        # Summary
        print(f"\n{'='*60}")
        print("MULTI-DATASET EVALUATION SUMMARY")
        print(f"{'='*60}")
        for ct, r in all_results.items():
            m = r['metrics']
            print(f"  [{ct}] AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                  f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
        print(f"\n  Average: AUROC={avg_metrics['auroc']:.4f} "
              f"AUPRC={avg_metrics['auprc']:.4f} F1={avg_metrics['f1']:.4f}")

        return combined_results

    else:
        # Single-dataset mode (backward compatible)
        data_path = os.path.join(config['data']['processed_data_dir'], 'graph_data.pt')
        data = torch.load(data_path, weights_only=False)

        gene_names_path = os.path.join(config['data']['processed_data_dir'], 'gene_names.pkl')
        try:
            with open(gene_names_path, 'rb') as f:
                gene_names = pickle.load(f)
        except FileNotFoundError:
            gene_names = [f'Gene_{i}' for i in range(data.num_nodes)]

        checkpoint_dir = os.path.dirname(checkpoint_path)
        experiment_dir = os.path.dirname(checkpoint_dir)
        splits_path = os.path.join(experiment_dir, 'edge_splits', 'edge_splits.pkl')
        if not os.path.exists(splits_path):
            splits_path = os.path.join(checkpoint_dir, 'edge_splits.pkl')

        if os.path.exists(splits_path):
            _, _, test_loader, _ = create_dataloaders(data, config, splits_path=splits_path)
        else:
            _, _, test_loader, _ = create_dataloaders(data, config)

        model = create_model(config, data.num_node_features)
        checkpoint = load_checkpoint(checkpoint_path, model)
        model = model.to(device)
        model.eval()

        if 'optimal_threshold' in checkpoint and checkpoint['optimal_threshold'] is not None:
            optimal_threshold = checkpoint['optimal_threshold']
        else:
            optimal_threshold = config['evaluation'].get('threshold', 0.5)

        return _evaluate_single_dataset(
            model, data, gene_names, test_loader,
            optimal_threshold, config, device, output_dir
        )


if __name__ == '__main__':
    main()
