# GeneInference: Unified Pipeline for Gene Regulatory Network Inference

A comprehensive Graph Neural Network (GNN) framework for inferring Gene Regulatory Networks (GRNs) from single-cell RNA-sequencing data.

## Overview

GeneInference formulates GRN inference as a link prediction task on gene co-expression graphs. The pipeline supports:

- **Multiple GNN Architectures**: Transformer, GAT, GIN, GraphSAGE, GCN, MLP
- **Automated Hyperparameter Sweeps**: Systematic exploration of model configurations
- **Experiment Tracking**: Centralized management of all experiments
- **Visualization Dashboard**: Comprehensive analysis of results
- **Complete Pipeline**: Download → Preprocess → Train → Evaluate → Predict

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Run complete pipeline (download → preprocess → train → evaluate → predict)
python main.py --experiment-type full
```

## Pipeline Architecture

```
GeneInference/
├── main.py                     # Single entry point for all experiments
├── config.yaml                 # Base configuration
├── sweep_config.yaml           # Hyperparameter sweep configuration
├── visualize_results.py        # Visualization dashboard
├── migrate_experiments.py      # Migration tool for old experiments
├── experiment_registry.csv     # Central experiment tracking
│
├── src/                        # Organized source code
│   ├── data/                   # Data handling
│   ├── models/                 # GNN architectures
│   ├── training/               # Training loop
│   ├── evaluation/             # Evaluation & visualization
│   └── utils/                  # Utilities & experiment manager
│
├── experiments/                # All experiments stored here
│   └── {experiment_id}/
│       ├── config.yaml
│       ├── checkpoints/
│       ├── edge_splits/
│       ├── results/
│       ├── predictions/
│       └── logs/
│
└── data/
    ├── raw/                    # Downloaded datasets
    └── processed/              # Preprocessed graphs
```

## Experiment Types

### 1. Download Data

```bash
python main.py --experiment-type download
```

Downloads datasets specified in `config.yaml`. Supported datasets:
- `beeline_hESC` - Human Embryonic Stem Cells (recommended for beginners)
- `beeline_mESC` - Mouse Embryonic Stem Cells
- `beeline_mDC` - Mouse Dendritic Cells
- `beeline_mHSC` - Mouse Hematopoietic Stem Cells
- `beeline_HepG2` - Human Hepatocellular Carcinoma

### 2. Preprocess Data

```bash
python main.py --experiment-type preprocess
```

Preprocesses raw expression data into PyTorch Geometric graph format:
- Filters genes by variance
- Computes co-expression network (Pearson correlation)
- Creates node features (mean, std, CV, isTF)
- Packages as PyTorch Geometric Data object

### 3. Train Model

```bash
python main.py --experiment-type train
```

Trains a GNN model with configuration from `config.yaml`:
- Creates train/val/test edge splits (70/15/15)
- Trains with early stopping
- Finds optimal classification threshold
- Saves best checkpoint

### 4. Evaluate Model

```bash
python main.py --experiment-type evaluate --eval-experiment-id <EXPERIMENT_ID>
```

Evaluates a trained model:
- Computes metrics: AUROC, AUPRC, Precision, Recall, F1
- Generates visualizations:
  - ROC curve
  - Precision-Recall curve
  - Confusion matrix
  - Threshold analysis
  - Per-TF performance

### 5. Predict GRN

```bash
python main.py --experiment-type predict --trained-experiment-id <EXPERIMENT_ID>
```

Generates GRN predictions:
- Predicts all possible TF→gene edges
- Filters by threshold
- Outputs 2D network visualization
- Saves predictions as CSV

### 6. Hyperparameter Sweep

```bash
python main.py --experiment-type sweep --sweep-config sweep_config.yaml
```

Runs systematic hyperparameter exploration:
- Tests all parameter combinations in `sweep_config.yaml`
- Tracks all results in registry
- Identifies best configuration

Example `sweep_config.yaml`:

```yaml
model.model_type:
  - transformer
  - gat

model.activation:
  - leaky_relu
  - gelu

model.hidden_dim:
  - 64
  - 128

model.num_layers:
  - 2
  - 3
```

### 7. Full Pipeline

```bash
python main.py --experiment-type full
```

Runs complete pipeline end-to-end:
1. Download dataset
2. Preprocess into graph
3. Train model
4. Evaluate performance
5. Generate GRN predictions

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  dataset_name: 'beeline_hESC'
  correlation_threshold: 0.3
  max_genes: 500

model:
  model_type: 'transformer'
  hidden_dim: 64
  num_layers: 2
  activation: 'leaky_relu'
  dropout: 0.2
  num_heads: 4

training:
  num_epochs: 1000
  batch_size: 256
  learning_rate: 0.0005
  patience: 100
```

## Experiment Management

### List All Experiments

```bash
python main.py --list-experiments
```

### Show Best Experiment

```bash
python main.py --best-experiment
```

### View Summary Statistics

```bash
python main.py --summary
```

### Resume Training

```bash
python main.py --experiment-type train --resume-from <EXPERIMENT_ID>
```

## Visualization Dashboard

Generate comprehensive visualizations of all experiments:

```bash
python visualize_results.py
```

Creates:
- `visualizations/metrics_distribution.png` - Distribution of all metrics
- `visualizations/model_comparison.png` - Performance by model type
- `visualizations/activation_comparison.png` - Performance by activation
- `visualizations/dataset_comparison.png` - Performance by dataset
- `visualizations/auroc_vs_auprc.png` - Scatter plot of AUROC vs AUPRC
- `visualizations/top10_experiments.png` - Top 10 experiments
- `visualizations/progress_over_time.png` - Training progress timeline
- `visualizations/summary_report.txt` - Text summary report

## Migrating Old Experiments

If you have experiments from the old structure:

```bash
# Dry run (see what will be migrated)
python migrate_experiments.py --dry-run

# Actually migrate
python migrate_experiments.py

# Clean up old experiments (keep only best 10)
python migrate_experiments.py --cleanup --keep-best 10 --dry-run
python migrate_experiments.py --cleanup --keep-best 10
```

## Model Architectures

### 1. Transformer (Best Performance)
- Multi-head self-attention on graph
- State-of-the-art expressiveness
- Recommended: `activation='leaky_relu'`, `num_heads=4`

### 2. GAT (Graph Attention Network)
- Multi-head attention mechanism
- Good interpretability
- Recommended: `activation='gelu'`, `num_heads=4`

### 3. GIN (Graph Isomorphism Network)
- MLP-based aggregation
- Provably expressive
- Recommended: `activation='relu'`

### 4. GraphSAGE
- Scalable to large graphs
- Fixed neighbor sampling
- Recommended: `activation='relu'`

### 5. GCN (Graph Convolutional Network)
- Simple graph convolution
- Good baseline
- Recommended: `activation='relu'`

### 6. MLP (Baseline)
- No graph structure
- Useful for comparison
- Recommended: `activation='leaky_relu'`

## Performance Metrics

Target metrics for GRN inference:
- **AUROC** > 0.90 (threshold-independent discrimination)
- **AUPRC** > 0.90 (precision-recall trade-off)
- **Precision** > 0.90 (minimize false positives)
- **Recall** = 1.0 (catch all true regulatory interactions)
- **F1** > 0.90 (harmonic mean)

## Advanced Usage

### Custom Notes

```bash
python main.py --experiment-type train --notes "Testing new architecture"
```

### Custom Config

```bash
python main.py --experiment-type train --config custom_config.yaml
```

### Programmatic Access

```python
from src.utils import ExperimentManager

# Initialize manager
exp_manager = ExperimentManager()

# Get best experiment
best_id = exp_manager.get_best_experiment(metric='auroc')

# Load experiment config
config = exp_manager.get_experiment_config(best_id)

# Compare experiments
comparison = exp_manager.compare_experiments([exp_id1, exp_id2, exp_id3])

# Export summary
exp_manager.export_summary('./my_summary.csv')
```

## Troubleshooting

### Out of Memory

Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 128  # Default is 256
```

### Training Too Slow

- Reduce `max_genes` in config
- Use smaller model: `hidden_dim: 32`, `num_layers: 2`
- Enable GPU: `training.use_gpu: true`

### Poor Performance

- Increase model capacity: `hidden_dim: 128`, `num_layers: 3`
- Try different activation: `gelu`, `silu`, `mish`
- Adjust correlation threshold: `data.correlation_threshold: 0.2`

## Directory Structure Details

### Experiments Directory

Each experiment creates:

```
experiments/20260204_120000_train_transformer_leaky_relu_h64_l2_beeline_hESC/
├── config.yaml              # Experiment configuration
├── checkpoints/
│   └── best_model.pt        # Trained model weights + threshold
├── edge_splits/
│   └── edge_splits.pkl      # Train/val/test edge split
├── results/
│   ├── results.yaml         # Metrics and results
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── ...
├── predictions/
│   └── predicted_grn.csv    # GRN predictions
└── logs/
    └── events.out.tfevents.* # TensorBoard logs
```

### Experiment Registry

`experiment_registry.csv` tracks all experiments:

| Column | Description |
|--------|-------------|
| experiment_id | Unique identifier |
| experiment_type | download/preprocess/train/evaluate/predict/sweep/full |
| dataset | Dataset name |
| model_type | GNN architecture |
| activation | Activation function |
| hidden_dim | Hidden dimension |
| num_layers | Number of GNN layers |
| created_at | Timestamp |
| status | created/running/completed/failed |
| auroc | Test AUROC |
| auprc | Test AUPRC |
| precision | Test Precision |
| recall | Test Recall |
| f1 | Test F1 Score |
| threshold | Optimal classification threshold |
| checkpoint_path | Path to checkpoints |
| results_path | Path to results |

## Citation

If you use GeneInference in your research, please cite:

```bibtex
@software{geneinference2026,
  title = {GeneInference: Unified Pipeline for Gene Regulatory Network Inference},
  author = {Kyler Zook},
  year = {2026},
  url = {https://github.com/yourusername/GeneInference}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- BEELINE benchmark: Pratapa et al. (2020) Nature Methods
- PyTorch Geometric: Fey & Lenssen (2019)
- Graph Neural Networks for biological networks
