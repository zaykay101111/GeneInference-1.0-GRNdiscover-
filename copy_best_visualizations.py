"""
Copy Best Model Visualizations to Central Folder
=================================================

This script copies visualizations from the best performing experiments
to a central visualizations/ folder with experiment IDs in the titles.

Author: Kyler Zook
Date: January 2026
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


def copy_best_visualizations(registry_path='experiment_registry.csv',
                            output_dir='visualizations',
                            top_n=5,
                            metric='auroc'):
    """
    Copy visualizations from best experiments to central folder.

    Args:
        registry_path: Path to experiment registry CSV
        output_dir: Output directory for copied visualizations
        top_n: Number of top experiments to copy
        metric: Metric to sort by (auroc, auprc, f1, etc.)
    """
    print("=" * 80)
    print("COPY BEST MODEL VISUALIZATIONS")
    print("=" * 80)

    # Load registry
    if not Path(registry_path).exists():
        print(f"\n✗ Registry not found: {registry_path}")
        print("  Run experiments first to create the registry.")
        return

    registry = pd.read_csv(registry_path)
    print(f"\nLoaded registry with {len(registry)} experiments")

    # Filter to completed experiments with valid metric
    completed = registry[
        (registry['status'] == 'completed') &
        (registry[metric].notna())
    ].copy()

    if len(completed) == 0:
        print(f"\n✗ No completed experiments with {metric} metric found")
        return

    print(f"Found {len(completed)} completed experiments with {metric} scores")

    # Sort by metric (descending)
    completed = completed.sort_values(metric, ascending=False)

    # Get top N
    top_experiments = completed.head(top_n)

    print(f"\nTop {top_n} experiments by {metric.upper()}:")
    for i, (_, row) in enumerate(top_experiments.iterrows(), 1):
        print(f"  {i}. {row['experiment_id']}: {metric.upper()}={row[metric]:.4f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define visualization types to copy
    viz_types = [
        'roc_curve.png',
        'pr_curve.png',
        'confusion_matrix.png',
        'prediction_heatmap.png',
        'threshold_analysis.png',
        'per_tf_performance.png'
    ]

    # Copy visualizations for each top experiment
    print(f"\nCopying visualizations to {output_dir}/...")

    copied_count = 0
    for rank, (_, row) in enumerate(top_experiments.iterrows(), 1):
        exp_id = row['experiment_id']
        results_path = row['results_path']

        if pd.isna(results_path) or not Path(results_path).exists():
            print(f"  ⚠ Skipping {exp_id}: results path not found")
            continue

        print(f"\n  [{rank}/{top_n}] Processing {exp_id}...")

        for viz_type in viz_types:
            source_path = Path(results_path) / viz_type

            if not source_path.exists():
                print(f"    ⊙ {viz_type}: not found")
                continue

            # Create new filename with experiment ID and rank
            viz_name = viz_type.replace('.png', '')
            new_filename = f"rank{rank}_{exp_id}_{viz_name}.png"
            dest_path = Path(output_dir) / new_filename

            # Copy file
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"    ✓ {viz_type} → {new_filename}")

    print(f"\n✓ Copied {copied_count} visualizations to {output_dir}/")

    # Create summary HTML page
    create_summary_html(top_experiments, output_dir, metric)

    print(f"✓ Created summary page: {output_dir}/index.html")

    print("\n" + "=" * 80)
    print("VISUALIZATION COPY COMPLETE")
    print("=" * 80)


def create_summary_html(top_experiments, output_dir, metric):
    """
    Create an HTML summary page with all visualizations.

    Args:
        top_experiments: DataFrame with top experiments
        output_dir: Output directory
        metric: Metric used for sorting
    """
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html>")
    html.append("<head>")
    html.append("  <title>GeneInference - Best Model Visualizations</title>")
    html.append("  <style>")
    html.append("    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }")
    html.append("    h1 { color: #000000; }")
    html.append("    h2 { color: #A23B72; margin-top: 30px; }")
    html.append("    .experiment { background-color: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html.append("    .metrics { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }")
    html.append("    .metrics table { width: 100%; border-collapse: collapse; }")
    html.append("    .metrics td { padding: 5px 10px; }")
    html.append("    .metrics td:first-child { font-weight: bold; width: 150px; }")
    html.append("    .viz-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }")
    html.append("    .viz-item { text-align: center; }")
    html.append("    .viz-item img { width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html.append("    .viz-item p { margin-top: 10px; font-weight: bold; color: #555; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("  <h1> GeneInference - Best Model Visualizations</h1>")
    html.append(f"  <p>Top {len(top_experiments)} experiments ranked by {metric.upper()}</p>")

    for rank, (_, row) in enumerate(top_experiments.iterrows(), 1):
        exp_id = row['experiment_id']

        html.append(f"  <div class='experiment'>")
        html.append(f"    <h2>Rank #{rank}: {exp_id}</h2>")

        # Metrics table
        html.append("    <div class='metrics'>")
        html.append("      <table>")
        html.append(f"        <tr><td>Model Type:</td><td>{row['model_type']}</td></tr>")
        html.append(f"        <tr><td>Activation:</td><td>{row['activation']}</td></tr>")
        html.append(f"        <tr><td>Hidden Dim:</td><td>{row['hidden_dim']}</td></tr>")
        html.append(f"        <tr><td>Num Layers:</td><td>{row['num_layers']}</td></tr>")
        html.append(f"        <tr><td><strong>AUROC:</strong></td><td><strong>{row['auroc']:.4f}</strong></td></tr>")
        html.append(f"        <tr><td><strong>AUPRC:</strong></td><td><strong>{row['auprc']:.4f}</strong></td></tr>")
        html.append(f"        <tr><td>Precision:</td><td>{row['precision']:.4f}</td></tr>")
        html.append(f"        <tr><td>Recall:</td><td>{row['recall']:.4f}</td></tr>")
        html.append(f"        <tr><td>F1 Score:</td><td>{row['f1']:.4f}</td></tr>")
        html.append("      </table>")
        html.append("    </div>")

        # Visualizations grid
        html.append("    <div class='viz-grid'>")

        viz_types = [
            ('roc_curve.png', 'ROC Curve'),
            ('pr_curve.png', 'Precision-Recall Curve'),
            ('confusion_matrix.png', 'Confusion Matrix'),
            ('prediction_heatmap.png', 'Prediction Heatmap'),
            ('threshold_analysis.png', 'Threshold Analysis'),
            ('per_tf_performance.png', 'Per-TF Performance')
        ]

        for viz_file, viz_title in viz_types:
            viz_name = viz_file.replace('.png', '')
            filename = f"rank{rank}_{exp_id}_{viz_name}.png"
            filepath = Path(output_dir) / filename

            if filepath.exists():
                html.append("      <div class='viz-item'>")
                html.append(f"        <img src='{filename}' alt='{viz_title}'>")
                html.append(f"        <p>{viz_title}</p>")
                html.append("      </div>")

        html.append("    </div>")
        html.append("  </div>")

    html.append("</body>")
    html.append("</html>")

    # Write HTML file
    with open(Path(output_dir) / 'index.html', 'w') as f:
        f.write('\n'.join(html))


def main():
    parser = argparse.ArgumentParser(
        description='Copy best model visualizations to central folder'
    )

    parser.add_argument('--registry', type=str, default='experiment_registry.csv',
                       help='Path to experiment registry')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of top experiments to copy')
    parser.add_argument('--metric', type=str, default='auroc',
                       choices=['auroc', 'auprc', 'f1', 'precision', 'recall'],
                       help='Metric to rank experiments by')

    args = parser.parse_args()

    copy_best_visualizations(
        registry_path=args.registry,
        output_dir=args.output_dir,
        top_n=args.top_n,
        metric=args.metric
    )


if __name__ == '__main__':
    main()
