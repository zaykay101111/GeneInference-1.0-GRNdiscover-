"""
Visualization Dashboard for GeneInference Experiments
Creates comprehensive visualizations of all experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np


def load_registry(registry_path='experiment_registry.csv'):
    """Load experiment registry"""
    if not Path(registry_path).exists():
        print(f"Registry not found: {registry_path}")
        return None

    return pd.read_csv(registry_path)


def plot_metric_comparison(df, output_dir='./visualizations'):
    """Plot comparison of metrics across experiments"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter completed experiments
    completed = df[df['status'] == 'completed'].copy()

    if len(completed) == 0:
        print("No completed experiments to visualize")
        return

    # Remove NaN values
    metrics = ['auroc', 'auprc', 'precision', 'recall', 'f1']
    for metric in metrics:
        completed = completed.dropna(subset=[metric])

    if len(completed) == 0:
        print("No experiments with metrics found")
        return

    # 1. Overall metrics distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Metrics Distribution Across All Experiments', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        completed[metric].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.axvline(completed[metric].mean(), color='red', linestyle='--',
                   label=f'Mean: {completed[metric].mean():.4f}')
        ax.legend()

    # Remove empty subplot
    axes[1, 2].remove()

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'metrics_distribution.png'}")
    plt.close()

    # 2. Model type comparison
    if 'model_type' in completed.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Performance by Model Type', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(['auroc', 'auprc', 'f1']):
            ax = axes[idx]
            model_metrics = completed.groupby('model_type')[metric].agg(['mean', 'std', 'count'])
            model_metrics = model_metrics.sort_values('mean', ascending=False)

            x = range(len(model_metrics))
            ax.bar(x, model_metrics['mean'], yerr=model_metrics['std'],
                   capsize=5, alpha=0.7, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(model_metrics.index, rotation=45, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} by Model Type')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (_, row) in enumerate(model_metrics.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'model_comparison.png'}")
        plt.close()

    # 3. Activation function comparison
    if 'activation' in completed.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Performance by Activation Function', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(['auroc', 'auprc', 'f1']):
            ax = axes[idx]
            act_metrics = completed.groupby('activation')[metric].agg(['mean', 'std', 'count'])
            act_metrics = act_metrics.sort_values('mean', ascending=False)

            x = range(len(act_metrics))
            ax.bar(x, act_metrics['mean'], yerr=act_metrics['std'],
                   capsize=5, alpha=0.7, edgecolor='black', color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(act_metrics.index, rotation=45, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} by Activation')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (_, row) in enumerate(act_metrics.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'activation_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'activation_comparison.png'}")
        plt.close()

    # 4. Dataset comparison
    if 'dataset' in completed.columns and completed['dataset'].nunique() > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Performance by Dataset', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(['auroc', 'auprc', 'f1']):
            ax = axes[idx]
            dataset_metrics = completed.groupby('dataset')[metric].agg(['mean', 'std', 'count'])
            dataset_metrics = dataset_metrics.sort_values('mean', ascending=False)

            x = range(len(dataset_metrics))
            ax.bar(x, dataset_metrics['mean'], yerr=dataset_metrics['std'],
                   capsize=5, alpha=0.7, edgecolor='black', color='green')
            ax.set_xticks(x)
            ax.set_xticklabels(dataset_metrics.index, rotation=45, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} by Dataset')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (_, row) in enumerate(dataset_metrics.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'dataset_comparison.png'}")
        plt.close()

    
    # Target thresholds
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (0.9)')
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_vs_auprc.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'auroc_vs_auprc.png'}")
    plt.close()

    # 6. Top 3 experiments
    fig, ax = plt.subplots(figsize=(14, 8))

    top3 = completed.nlargest(3, 'auroc')[['experiment_id', 'auroc', 'auprc', 'f1']].copy()
    top3['short_id'] = top10['experiment_id'].str[:40] + '...'

    x = range(len(top10))
    width = 0.25

    ax.bar([i - width for i in x], top10['auroc'], width, label='AUROC', alpha=0.8, color='#676767')
    ax.bar(x, top10['auprc'], width, label='AUPRC', alpha=0.8,color='#B2BEB5')
    ax.bar([i + width for i in x], top10['f1'], width, label='F1', alpha=0.8, color='#808080')

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Top 3 Experiments by AUROC', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top10['short_id'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'top10_experiments.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'top10_experiments.png'}")
    plt.close()


def plot_training_progress(df, output_dir='./visualizations'):
    """Plot training progress over time"""
    output_dir = Path(output_dir)

    # Filter completed experiments with valid created_at
    completed = df[(df['status'] == 'completed') & df['created_at'].notna()].copy()

    if len(completed) == 0:
        return

    # Convert created_at to datetime
    completed['created_at'] = pd.to_datetime(completed['created_at'], format='%Y%m%d_%H%M%S')
    completed = completed.sort_values('created_at')

    # Calculate cumulative best metrics
    completed['best_auroc_so_far'] = completed['auroc'].cummax()
    completed['best_auprc_so_far'] = completed['auprc'].cummax()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Progress Over Time', fontsize=16, fontweight='bold')

    # AUROC over time
    axes[0].plot(completed['created_at'], completed['auroc'],
                'o-', alpha=0.5, label='Individual Experiments')
    axes[0].plot(completed['created_at'], completed['best_auroc_so_far'],
                'r-', linewidth=2, label='Best So Far')
    axes[0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.9)')
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('AUROC Progress')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUPRC over time
    axes[1].plot(completed['created_at'], completed['auprc'],
                'o-', alpha=0.5, label='Individual Experiments')
    axes[1].plot(completed['created_at'], completed['best_auprc_so_far'],
                'r-', linewidth=2, label='Best So Far')
    axes[1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.9)')
    axes[1].set_ylabel('AUPRC', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_title('AUPRC Progress')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'progress_over_time.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'progress_over_time.png'}")
    plt.close()


def generate_summary_report(df, output_path='./visualizations/summary_report.txt'):
    """Generate text summary report"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = df[df['status'] == 'completed']

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GENEINFERENCE EXPERIMENT SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"Completed: {len(completed)}\n")
        f.write(f"Running: {len(df[df['status'] == 'running'])}\n")
        f.write(f"Failed: {len(df[df['status'] == 'failed'])}\n")
        f.write(f"Created: {len(df[df['status'] == 'created'])}\n\n")

        if len(completed) > 0:
            # Best metrics
            f.write("BEST METRICS (All Experiments)\n")
            f.write("-"*80 + "\n")
            f.write(f"Best AUROC: {completed['auroc'].max():.4f}\n")
            f.write(f"Best AUPRC: {completed['auprc'].max():.4f}\n")
            f.write(f"Best F1: {completed['f1'].max():.4f}\n")
            f.write(f"Best Precision: {completed['precision'].max():.4f}\n")
            f.write(f"Best Recall: {completed['recall'].max():.4f}\n\n")

            # Average metrics
            f.write("AVERAGE METRICS (All Experiments)\n")
            f.write("-"*80 + "\n")
            f.write(f"Avg AUROC: {completed['auroc'].mean():.4f} ± {completed['auroc'].std():.4f}\n")
            f.write(f"Avg AUPRC: {completed['auprc'].mean():.4f} ± {completed['auprc'].std():.4f}\n")
            f.write(f"Avg F1: {completed['f1'].mean():.4f} ± {completed['f1'].std():.4f}\n\n")

            # Best experiment
            best_idx = completed['auroc'].idxmax()
            best_exp = completed.loc[best_idx]

            f.write("BEST EXPERIMENT (by AUROC)\n")
            f.write("-"*80 + "\n")
            f.write(f"Experiment ID: {best_exp['experiment_id']}\n")
            f.write(f"Model Type: {best_exp['model_type']}\n")
            f.write(f"Activation: {best_exp['activation']}\n")
            f.write(f"Dataset: {best_exp['dataset']}\n")
            f.write(f"AUROC: {best_exp['auroc']:.4f}\n")
            f.write(f"AUPRC: {best_exp['auprc']:.4f}\n")
            f.write(f"Precision: {best_exp['precision']:.4f}\n")
            f.write(f"Recall: {best_exp['recall']:.4f}\n")
            f.write(f"F1: {best_exp['f1']:.4f}\n\n")

            # Top 5 experiments
            f.write("TOP 5 EXPERIMENTS (by AUROC)\n")
            f.write("-"*80 + "\n")
            top5 = completed.nlargest(5, 'auroc')
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                f.write(f"\n{i}. {row['experiment_id'][:60]}\n")
                f.write(f"   AUROC: {row['auroc']:.4f}, AUPRC: {row['auprc']:.4f}, F1: {row['f1']:.4f}\n")
                f.write(f"   Model: {row['model_type']}, Activation: {row['activation']}\n")

            # Experiments meeting target
            meets_target = completed[(completed['auroc'] >= 0.9) &
                                    (completed['auprc'] >= 0.9) &
                                    (completed['precision'] >= 0.9)]

            f.write(f"\n\nEXPERIMENTS MEETING TARGET (AUROC, AUPRC, Precision >= 0.9)\n")
            f.write("-"*80 + "\n")
            f.write(f"Count: {len(meets_target)}\n")
            if len(meets_target) > 0:
                for _, row in meets_target.iterrows():
                    f.write(f"\n- {row['experiment_id'][:60]}\n")
                    f.write(f"  AUROC: {row['auroc']:.4f}, AUPRC: {row['auprc']:.4f}, "
                          f"Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}\n")

    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GeneInference experiment results')
    parser.add_argument('--registry', type=str, default='experiment_registry.csv',
                       help='Path to experiment registry CSV')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Load registry
    print("\nLoading experiment registry...")
    df = load_registry(args.registry)

    if df is None or len(df) == 0:
        print("No experiments found")
        return

    print(f"Found {len(df)} experiments")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metric_comparison(df, args.output_dir)
    plot_training_progress(df, args.output_dir)
    generate_summary_report(df, Path(args.output_dir) / 'summary_report.txt')

    print(f"\n✓ All visualizations saved to: {args.output_dir}")
    print(f"\nTo view results:")
    print(f"  open {args.output_dir}")


if __name__ == '__main__':
    main()
