"""
Training Script for SimpleGRN
==============================

This script trains the GRN inference model using the processed data.

Training loop:
1. Load processed data and create dataloaders
2. Initialize model, optimizer, and loss function
3. For each epoch:
   - Train on training set
   - Evaluate on validation set
   - Save best model
4. Load best model and evaluate on test set

Author: [Your Name]
Date: [Date]
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import pickle
from pathlib import Path
import time
from tqdm import tqdm

# Import project modules
import sys
from pathlib import Path
# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.architectures import create_model
from src.data.dataset import (
    create_dataloaders,
    load_single_dataset,
    load_multiple_datasets,
    create_multi_dataloaders,
)
from src.utils.helpers import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    compute_metrics,
    find_optimal_threshold
)


class AdaptivePrecisionRecallLoss(nn.Module):
    """
    Adaptive loss that dynamically balances precision and recall.

    Key insight: To achieve both high recall AND high precision, we need:
    1. Start with balanced weights to establish good precision
    2. Adaptively increase FN weight only if recall drops
    3. Use focal loss to focus on hard examples (especially hard negatives)

    The weights adapt during training based on current metrics.
    """
    def __init__(self, initial_fn_weight=25.0, initial_fp_weight=25.0, gamma=2.0):
        super(AdaptivePrecisionRecallLoss, self).__init__()
        self.fn_weight = initial_fn_weight
        self.fp_weight = initial_fp_weight
        self.gamma = gamma
        self.target_recall = 1.0
        self.target_precision = 0.90

    def update_weights(self, current_recall, current_precision):
        """
        Dynamically adjust weights based on current metrics.
        Called after each epoch to adapt the loss function.

        Strategy:
        1. If recall is very low (<0.5), strongly increase FN weight
        2. If recall is moderate but below target, gently increase FN weight
        3. If recall is good but precision is low, increase FP weight
        """
        # Strong adjustment if recall is very poor
        if current_recall < 0.5:
            self.fn_weight *= 1.05
        # Gentle adjustment if recall is moderate
        elif current_recall < self.target_recall - 0.1:
            self.fn_weight *= 1.02
        # If recall is good but precision needs improvement
        elif current_recall >= self.target_recall - 0.1 and current_precision < self.target_precision:
            self.fp_weight *= 1.02

        # Cap weights to prevent instability
        self.fn_weight = min(self.fn_weight, 80.0)
        self.fp_weight = min(self.fp_weight, 80.0)

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Focal loss component: down-weight easy examples
        # This helps focus on hard negatives (high correlation but no regulation)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute weighted focal binary cross entropy
        pos_loss = -targets * torch.log(probs + 1e-7) * focal_weight * self.fn_weight
        neg_loss = -(1 - targets) * torch.log(1 - probs + 1e-7) * focal_weight * self.fp_weight

        loss = pos_loss + neg_loss
        return loss.mean()


class RecallPrecisionLoss(nn.Module):
    """
    Custom loss that balances recall=1.0 and high precision
    Uses focal loss component to handle easy negatives better

    NOTE: This is the legacy version. Use AdaptivePrecisionRecallLoss for better results.
    """
    def __init__(self, false_negative_weight=200.0, false_positive_weight=10.0, gamma=2.0):
        super(RecallPrecisionLoss, self).__init__()
        self.fn_weight = false_negative_weight
        self.fp_weight = false_positive_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Focal loss component: down-weight easy examples
        # For positives: focus on hard-to-classify positives
        # For negatives: focus on hard negatives (high false positive risk)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute weighted focal binary cross entropy
        # Heavily penalize missing positives (FN), moderately penalize false positives (FP)
        pos_loss = -targets * torch.log(probs + 1e-7) * focal_weight * self.fn_weight
        neg_loss = -(1 - targets) * torch.log(1 - probs + 1e-7) * focal_weight * self.fp_weight

        loss = pos_loss + neg_loss
        return loss.mean()


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, data, train_loader, optimizer, criterion, device, config,
                data_list=None):
    """
    Train model for one epoch.

    Supports both single-dataset and multi-dataset modes:
    - Single-dataset: data is a Data object, batches have no 'dataset_idx'
    - Multi-dataset: data_list is a list of Data objects, batches have 'dataset_idx'

    Args:
        model: GNN model
        data: PyTorch Geometric Data object (single-dataset mode)
        train_loader: DataLoader for training edges
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        config: Configuration dict
        data_list: list of Data objects indexed by dataset_idx (multi-dataset mode)

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    multi_dataset = data_list is not None

    if not multi_dataset:
        # Single-dataset: pre-move to device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

    # Training loop
    for batch in train_loader:
        labels = batch['labels'].to(device)
        optimizer.zero_grad()

        if multi_dataset and 'dataset_idx' in batch:
            # Multi-dataset: route edges through correct graphs
            ds_indices = batch['dataset_idx'].to(device)
            edges = batch['edges'].to(device)
            correlations = batch['correlations'].to(device)

            all_predictions = torch.zeros(len(labels), device=device)

            for ds_idx in ds_indices.unique():
                mask = ds_indices == ds_idx
                ds_edges = edges[:, mask]
                ds_corr = correlations[mask]

                ds_data = data_list[ds_idx.item()]
                ds_x = ds_data.x.to(device)
                ds_edge_index = ds_data.edge_index.to(device)

                preds = model(ds_x, ds_edge_index, ds_edges, ds_corr)
                all_predictions[mask] = preds

            predictions = all_predictions
        else:
            # Single-dataset
            edges = batch['edges'].to(device)
            correlations = batch['correlations'].to(device)
            predictions = model(x, edge_index, edges, correlations)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, data, val_loader, criterion, device, config,
                   threshold=None, find_threshold=False, quiet=True,
                   data_list=None):
    """
    Evaluate model on validation set.

    Supports both single-dataset and multi-dataset modes.

    Args:
        model: GNN model
        data: PyTorch Geometric Data object (single-dataset mode)
        val_loader: DataLoader for validation edges
        criterion: Loss function
        device: Device to use
        config: Configuration dict
        threshold: Classification threshold (if None, uses 0.5)
        find_threshold: If True, finds optimal threshold and returns it
        quiet: If True, suppress output messages
        data_list: list of Data objects indexed by dataset_idx (multi-dataset mode)

    Returns:
        tuple: (avg_loss, metrics_dict) or (avg_loss, metrics_dict, optimal_threshold) if find_threshold=True
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []

    multi_dataset = data_list is not None

    if not multi_dataset:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

    with torch.no_grad():
        for batch in val_loader:
            labels = batch['labels'].to(device)

            if multi_dataset and 'dataset_idx' in batch:
                ds_indices = batch['dataset_idx'].to(device)
                edges = batch['edges'].to(device)
                correlations = batch['correlations'].to(device)

                predictions = torch.zeros(len(labels), device=device)
                for ds_idx in ds_indices.unique():
                    mask = ds_indices == ds_idx
                    ds_edges = edges[:, mask]
                    ds_corr = correlations[mask]

                    ds_data = data_list[ds_idx.item()]
                    ds_x = ds_data.x.to(device)
                    ds_edge_index = ds_data.edge_index.to(device)

                    preds = model(ds_x, ds_edge_index, ds_edges, ds_corr)
                    predictions[mask] = preds
            else:
                edges = batch['edges'].to(device)
                correlations = batch['correlations'].to(device)
                predictions = model(x, edge_index, edges, correlations)

            loss = criterion(predictions, labels)

            total_loss += loss.item()
            num_batches += 1
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / num_batches
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Apply sigmoid to convert logits to probabilities for metrics
    all_predictions = torch.sigmoid(all_predictions)

    # Find optimal threshold if requested
    if find_threshold:
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            all_predictions, all_labels, metric='f1'
        )
        if not quiet:
            print(f"\n  Optimal threshold: {optimal_threshold:.4f}")
            print(f"    Precision: {threshold_metrics['precision']:.4f}")
            print(f"    Recall: {threshold_metrics['recall']:.4f}")
            print(f"    F1: {threshold_metrics['f1']:.4f}")

        metrics = compute_metrics(
            all_predictions,
            all_labels,
            config['evaluation']['metrics'],
            threshold=optimal_threshold
        )
        return avg_loss, metrics, optimal_threshold
    else:
        if threshold is None:
            threshold = 0.5
        metrics = compute_metrics(
            all_predictions,
            all_labels,
            config['evaluation']['metrics'],
            threshold=threshold
        )
        return avg_loss, metrics


def train(config, quiet=True):
    """
    Main training function.

    This orchestrates the entire training process:
    1. Setup (device, logging, directories)
    2. Load data
    3. Create model
    4. Training loop with validation
    5. Save best model
    6. Final evaluation on test set

    Args:
        config (dict): Configuration dictionary
        quiet (bool): If True, suppress epoch-by-epoch output and show progress bar instead
    """
    if not quiet:
        print("="*60)
        print("SimpleGRN Training")
        print("="*60)

    # ==================================================
    # Setup
    # ==================================================

    # Set random seed for reproducibility
    set_seed(config['training']['seed'])

    # Setup device (GPU if available)
    if torch.cuda.is_available() and config['training']['use_gpu']:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and config['training']['use_gpu']:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    #if not quiet:
    print(f"\nUsing device: {device}")

    # Create output directories
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)

    # Setup TensorBoard logging
    writer = SummaryWriter(config['output']['log_dir'])

    # ==================================================
    # Load Data (multi-dataset or single-dataset)
    # ==================================================

    if not quiet:
        print("\nLoading processed data...")

    # Determine if multi-dataset mode
    use_multi = ('datasets' in config['data'] and
                 'train' in config['data']['datasets'] and
                 len(config['data']['datasets']['train']) > 0)

    if '_edge_splits_path' in config:
        splits_path = config['_edge_splits_path']
    else:
        splits_path = os.path.join(config['output']['checkpoint_dir'], 'edge_splits.pkl')

    if use_multi:
        # Multi-dataset mode
        print("\n  Loading training datasets...")
        train_datasets = load_multiple_datasets(config, split='train')
        print("\n  Loading test datasets...")
        test_datasets = load_multiple_datasets(config, split='test')

        # Extract Data objects
        train_data = {ct: data for ct, (data, _) in train_datasets.items()}
        test_data = {ct: data for ct, (data, _) in test_datasets.items()}

        # Create multi-dataset loaders
        (train_loader, val_loader, test_loaders_dict,
         data_list, cell_type_to_idx) = create_multi_dataloaders(
            train_data, test_data, config, splits_path=splits_path
        )

        # For model creation, use first dataset's feature count (all should be same)
        first_data = data_list[0]
        num_node_features = first_data.num_node_features

        # data is set to None for single-dataset functions (not used in multi mode)
        data = None
        test_loader = None  # replaced by test_loaders_dict
    else:
        # Single-dataset mode (backward compatible)
        data_path = os.path.join(config['data']['processed_data_dir'], 'graph_data.pt')
        data = torch.load(data_path, weights_only=False)
        if not quiet:
            print(f"  Nodes: {data.num_nodes}")
            print(f"  Edges: {data.num_edges}")
            print(f"  Features: {data.num_node_features}")

        train_loader, val_loader, test_loader, edge_splits = create_dataloaders(
            data, config, splits_path=splits_path
        )
        num_node_features = data.num_node_features
        data_list = None
        test_loaders_dict = None
        cell_type_to_idx = None

    # ==================================================
    # Create Model
    # ==================================================

    if not quiet:
        print("\nInitializing model...")

    # Create model
    model = create_model(config, num_node_features)
    model = model.to(device)

    # Print model architecture
    if not quiet:
        print(model)

    # ==================================================
    # Setup Training
    # ==================================================

    # Define loss function
    # Use the proven RecallPrecisionLoss with parameters that achieved good results
    # FN weight >> FP weight ensures high recall
    # Focal loss (gamma=2.0) helps with hard examples
    criterion = RecallPrecisionLoss(
        false_negative_weight=50.0,
        false_positive_weight=20.0,
        gamma=2.0
    )

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Define learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['training']['patience'] // 2
    )

    # ==================================================
    # Training Loop
    # ==================================================

    if not quiet:
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)

    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Get model name for progress bar description
    model_name = f"{config['model']['model_type']}_{config['model']['activation']}"

    # Training loop with progress bar
    progress_bar = tqdm(
        range(config['training']['num_epochs']),
        desc=f"Training {model_name}",
        disable=not quiet,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    for epoch in progress_bar:
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, data, train_loader, optimizer, criterion, device, config,
            data_list=data_list
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, data, val_loader, criterion, device, config, quiet=quiet,
            data_list=data_list
        )
        val_losses.append(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)

        # Update progress bar with current metrics
        if quiet:
            progress_bar.set_postfix({
                'loss': f"{val_loss:.4f}",
                'auroc': f"{val_metrics.get('auroc', 0):.3f}",
                'recall': f"{val_metrics.get('recall', 0):.3f}"
            })

        # Print progress (only if not quiet)
        if not quiet:
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            for metric_name, metric_value in val_metrics.items():
                print(f"  Val {metric_name}: {metric_value:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

        # Log loss weights periodically (if using adaptive loss)
        if hasattr(criterion, 'update_weights') and epoch % 50 == 0:
            current_recall = val_metrics.get('recall', 0.0)
            current_precision = val_metrics.get('precision', 0.0)
            criterion.update_weights(current_recall, current_precision)
            if not quiet:
                print(f"  Loss weights: FN={criterion.fn_weight:.1f}, FP={criterion.fp_weight:.1f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(config['output']['checkpoint_dir'], 'best_model.pt')
            )
            if not quiet:
                print("  ✓ Saved best model")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['training']['patience']:
            if not quiet:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        # Update learning rate
        scheduler.step(val_loss)

    progress_bar.close()

    # ==================================================
    # Threshold Optimization
    # ==================================================

    if not quiet:
        print("\n" + "="*60)
        print("Finding optimal classification threshold...")
        print("="*60)

    # Load best model
    checkpoint = load_checkpoint(
        os.path.join(config['output']['checkpoint_dir'], 'best_model.pt'),
        model,
        optimizer
    )
    if not quiet:
        print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Find optimal threshold on validation set
    if not quiet:
        print("\nOptimizing threshold on validation set...")
    val_loss, val_metrics, optimal_threshold = validate_epoch(
        model, data, val_loader, criterion, device, config,
        find_threshold=True, quiet=quiet, data_list=data_list
    )

    # Additionally check if we can get perfect recall with good precision
    from src.utils.helpers import find_optimal_threshold_constrained
    model.eval()
    all_preds = []
    all_labels = []

    if data_list is None:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

    with torch.no_grad():
        for batch in val_loader:
            labels = batch['labels'].to(device)

            if data_list is not None and 'dataset_idx' in batch:
                ds_indices = batch['dataset_idx'].to(device)
                edges = batch['edges'].to(device)
                correlations = batch['correlations'].to(device)

                preds = torch.zeros(len(labels), device=device)
                for ds_idx in ds_indices.unique():
                    mask = ds_indices == ds_idx
                    ds_data = data_list[ds_idx.item()]
                    ds_x = ds_data.x.to(device)
                    ds_ei = ds_data.edge_index.to(device)
                    p = model(ds_x, ds_ei, edges[:, mask], correlations[mask])
                    preds[mask] = p
            else:
                edges = batch['edges'].to(device)
                correlations = batch['correlations'].to(device)
                preds = model(x, edge_index, edges, correlations)

            preds = torch.sigmoid(preds)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Find threshold that gives perfect recall if possible
    constrained_threshold, constrained_precision, constrained_recall = find_optimal_threshold_constrained(
        all_preds, all_labels, min_recall=0.99
    )

    if not quiet:
        print(f"\n  Constrained threshold (recall >= 0.99): {constrained_threshold:.4f}")
        print(f"    Precision: {constrained_precision:.4f}")
        print(f"    Recall: {constrained_recall:.4f}")

    # Use constrained threshold if it gives better recall with acceptable precision
    if constrained_recall >= 0.99 and constrained_precision >= 0.85:
        optimal_threshold = constrained_threshold
        if not quiet:
            print(f"\n  Using constrained threshold for better recall")

    # Re-save checkpoint with optimal threshold
    save_checkpoint(
        model,
        optimizer,
        checkpoint['epoch'],
        checkpoint['loss'],
        os.path.join(config['output']['checkpoint_dir'], 'best_model.pt'),
        optimal_threshold=optimal_threshold
    )
    if not quiet:
        print(f"\n✓ Saved optimal threshold ({optimal_threshold:.4f}) with checkpoint")

    # ==================================================
    # Final Evaluation
    # ==================================================

    if not quiet:
        print("\n" + "="*60)
        print("Final evaluation on test set...")
        print("="*60)

    # Evaluate on test set(s) using optimal threshold
    if test_loaders_dict is not None:
        # Multi-dataset: evaluate each test dataset separately
        per_dataset_metrics = {}
        for cell_type, t_loader in test_loaders_dict.items():
            ct_idx = cell_type_to_idx[cell_type]
            ct_data = data_list[ct_idx]
            t_loss, t_metrics = validate_epoch(
                model, ct_data, t_loader, criterion, device, config,
                threshold=optimal_threshold, quiet=quiet
            )
            per_dataset_metrics[cell_type] = {k: float(v) for k, v in t_metrics.items()}
            per_dataset_metrics[cell_type]['test_loss'] = float(t_loss)

            if not quiet:
                print(f"\n  Test [{cell_type}]:")
                print(f"    Loss: {t_loss:.4f}")
                for mn, mv in t_metrics.items():
                    print(f"    {mn}: {mv:.4f}")

        # Use average metrics across test datasets as summary
        all_metric_names = list(next(iter(per_dataset_metrics.values())).keys())
        avg_metrics = {}
        for mn in all_metric_names:
            vals = [m[mn] for m in per_dataset_metrics.values() if mn in m]
            avg_metrics[mn] = sum(vals) / len(vals) if vals else 0.0

        test_loss = avg_metrics.get('test_loss', 0.0)
        test_metrics = {k: v for k, v in avg_metrics.items() if k != 'test_loss'}

        if not quiet:
            print(f"\n  Average Test Metrics:")
            for mn, mv in test_metrics.items():
                print(f"    {mn}: {mv:.4f}")

        results = {
            'test_loss': float(test_loss),
            'test_metrics': test_metrics,
            'per_dataset_metrics': per_dataset_metrics,
            'best_epoch': int(checkpoint['epoch']),
            'best_val_loss': float(checkpoint['loss']),
            'optimal_threshold': float(optimal_threshold)
        }
    else:
        # Single-dataset
        test_loss, test_metrics = validate_epoch(
            model, data, test_loader, criterion, device, config,
            threshold=optimal_threshold, quiet=quiet
        )

        if not quiet:
            print("\nTest Results (with optimal threshold):")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Threshold: {optimal_threshold:.4f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

        results = {
            'test_loss': float(test_loss),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'best_epoch': int(checkpoint['epoch']),
            'best_val_loss': float(checkpoint['loss']),
            'optimal_threshold': float(optimal_threshold)
        }

    # Save test results
    results_path = os.path.join(config['output']['results_dir'], 'test_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    if not quiet:
        print(f"\nSaved results to {results_path}")

    # Close TensorBoard writer
    writer.close()

    if not quiet:
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. View training curves: tensorboard --logdir", config['output']['log_dir'])
        print("2. Run evaluation: python evaluate.py")
        print("3. Make predictions: python predict.py")


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='Train SimpleGRN model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Train model
    train(config)


if __name__ == '__main__':
    main()
