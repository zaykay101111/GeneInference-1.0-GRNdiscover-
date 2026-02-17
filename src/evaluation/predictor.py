"""
Prediction Script for SimpleGRN
================================

Make predictions on new data using trained model.

Author: [Your Name]
Date: [Date]
"""

import torch
import pandas as pd
import yaml
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.architectures import create_model
from src.utils.helpers import load_checkpoint


def predict_grn(model, data, gene_names, top_k=1000, threshold=0.5):
    """
    Predict GRN for all gene pairs.

    Generates predictions for all possible gene pairs and returns
    the top K predictions ranked by confidence.

    Args:
        model: Trained GNN model
        data: PyTorch Geometric Data object with graph structure
        gene_names: List of gene names
        top_k: Number of top predictions to return (default: 1000)
        threshold: Probability threshold for predictions (default: 0.5)

    Returns:
        DataFrame with columns: source_gene, target_gene, probability, predicted_regulatory
    """
    print("\nGenerating GRN predictions...")

    device = next(model.parameters()).device
    model.eval()

    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Get TF indices (genes with is_TF=1, which is feature index 3)
    tf_indices = torch.where(data.x[:, 3] == 1.0)[0]
    num_genes = len(gene_names)

    print(f"  Total genes: {num_genes}")
    print(f"  Transcription factors: {len(tf_indices)}")
    print(f"  Generating predictions for all TF->gene pairs...")

    # Generate all TF->gene pairs
    all_predictions = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 10000

        for tf_idx in tf_indices:
            # Create edges from this TF to all genes (including itself)
            target_genes = torch.arange(num_genes, device=device)
            source_genes = torch.full((num_genes,), tf_idx.item(), device=device)

            # Create edge batch
            edges = torch.stack([source_genes, target_genes], dim=0)

            # Get actual correlations from the correlation matrix
            if hasattr(data, 'correlation_matrix') and data.correlation_matrix is not None:
                correlations = data.correlation_matrix[tf_idx].to(device)
            else:
                correlations = torch.zeros(num_genes, device=device)

            # Process in batches
            for i in range(0, num_genes, batch_size):
                end_idx = min(i + batch_size, num_genes)
                batch_edges = edges[:, i:end_idx]
                batch_corr = correlations[i:end_idx]  # Keep as 1D tensor

                # Get predictions
                logits = model(x, edge_index, batch_edges, batch_corr)
                probs = torch.sigmoid(logits).squeeze()

                # Convert to numpy and store
                for j, prob in enumerate(probs):
                    gene_idx = i + j
                    all_predictions.append({
                        'source_gene': gene_names[tf_idx],
                        'target_gene': gene_names[gene_idx],
                        'source_idx': tf_idx.item(),
                        'target_idx': gene_idx,
                        'probability': prob.item()
                    })

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)

    # Add predicted regulatory column
    predictions_df['predicted_regulatory'] = predictions_df['probability'] >= threshold

    # Sort by probability (descending)
    predictions_df = predictions_df.sort_values('probability', ascending=False)

    # Filter to top K
    if top_k is not None and len(predictions_df) > top_k:
        predictions_df = predictions_df.head(top_k)

    print(f"\n  Total predictions generated: {len(all_predictions)}")
    print(f"  Predictions above threshold ({threshold}): {predictions_df['predicted_regulatory'].sum()}")
    print(f"  Returning top {len(predictions_df)} predictions")

    return predictions_df


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict GRN using SimpleGRN')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()

    print("="*60)
    print("SimpleGRN Prediction")
    print("="*60)

    # TODO: Load model and make predictions
    print("[!] TODO: Implement prediction pipeline")


if __name__ == '__main__':
    main()
