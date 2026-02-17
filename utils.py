"""
Utility Functions for SimpleGRN
================================

Helper functions used throughout the project.

Author: [Your Name]
Date: [Date]
"""

import torch
import numpy as np
import random
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve
)


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    This ensures experiments can be reproduced.
    Important for research and debugging!

    TODO: Set all random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def save_checkpoint(model, optimizer, epoch, loss, filepath, optimal_threshold=None):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        optimal_threshold: Optimal classification threshold (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'optimal_threshold': optimal_threshold
    }
    torch.save(checkpoint, filepath)



def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    """

    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
    


def find_optimal_threshold(predictions, labels, metric='f1'):
    """
    Find optimal classification threshold for imbalanced datasets.

    For imbalanced datasets, the default threshold of 0.5 is often suboptimal.
    This function finds the threshold that maximizes a specified metric.

    Args:
        predictions: Model predictions (probabilities)
        labels: True labels
        metric: Metric to optimize ('f1', 'f1_weighted', 'youden', or 'precision_at_recall')
                - 'f1': Maximize F1 score (harmonic mean of precision/recall)
                - 'f1_weighted': F1 with 2x weight on recall (prioritize catching positives)
                - 'youden': Youden's J statistic (sensitivity + specificity - 1)
                - 'precision_at_recall': Maximize precision while maintaining recall >= 0.99

    Returns:
        tuple: (optimal_threshold, threshold_metrics)
    """
    # Convert to numpy
    pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    label_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Get precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(label_np, pred_np)

    # Compute F1 scores for each threshold
    # Note: precision_recall_curve returns n+1 precision/recall values but only n thresholds
    # The last values correspond to threshold=0 (predict everything as positive)
    # We use [:-1] to align arrays
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if metric == 'f1':
        # Standard F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

    elif metric == 'f1_weighted':
        # Weighted F1 with higher weight on recall (don't miss positives)
        # F_beta with beta=2 gives recall 2x the weight of precision
        beta = 2.0
        f_beta = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f_beta)
        optimal_threshold = thresholds[optimal_idx]

    elif metric == 'precision_at_recall':
        # Find threshold that maximizes precision while maintaining near-perfect recall
        # This is key for achieving high precision with recall = 1.0
        min_recall = 0.99
        best_precision = 0.0
        optimal_threshold = 0.5

        for i, (p, r, t) in enumerate(zip(precisions, recalls, thresholds)):
            if r >= min_recall and p > best_precision:
                best_precision = p
                optimal_threshold = t

        # If no threshold gives required recall, use lowest threshold
        if best_precision == 0.0:
            optimal_threshold = thresholds[np.argmax(recalls)]

    elif metric == 'youden':
        # Youden's J statistic = sensitivity + specificity - 1
        # Maximizes the difference between true positive rate and false positive rate
        # Requires computing specificity, which needs true negatives
        from sklearn.metrics import confusion_matrix
        youdens_j = []
        for threshold in thresholds:
            pred_binary = (pred_np >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(label_np, pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youdens_j.append(sensitivity + specificity - 1)
        optimal_idx = np.argmax(youdens_j)
        optimal_threshold = thresholds[optimal_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute metrics at optimal threshold
    pred_binary = (pred_np >= optimal_threshold).astype(int)
    threshold_metrics = {
        'precision': precision_score(label_np, pred_binary),
        'recall': recall_score(label_np, pred_binary),
        'f1': f1_score(label_np, pred_binary),
        'accuracy': accuracy_score(label_np, pred_binary)
    }

    return optimal_threshold, threshold_metrics


def find_optimal_threshold_constrained(predictions, labels, min_recall=0.99, target_precision=0.90):
    """
    Find optimal threshold that maximizes precision while maintaining minimum recall.

    This is specifically designed for achieving the target metrics:
    - Recall >= min_recall (default 0.99, effectively 1.0 for small test sets)
    - Precision as high as possible, ideally >= target_precision

    Args:
        predictions: Model predictions (probabilities)
        labels: True labels
        min_recall: Minimum acceptable recall (default 0.99)
        target_precision: Target precision to achieve (default 0.90)

    Returns:
        tuple: (optimal_threshold, achieved_precision, achieved_recall)
    """
    pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    label_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Test many thresholds
    thresholds = np.linspace(0.001, 0.999, 1000)
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        pred_binary = (pred_np >= threshold).astype(int)

        # Skip if no positive predictions
        if pred_binary.sum() == 0:
            continue

        recall = recall_score(label_np, pred_binary, zero_division=0)
        precision = precision_score(label_np, pred_binary, zero_division=0)

        # Only consider thresholds that maintain required recall
        if recall >= min_recall:
            if precision > best_precision:
                best_precision = precision
                best_threshold = threshold
                best_recall = recall

    return best_threshold, best_precision, best_recall


def apply_network_constraints(predictions, edges, node_features, gene_names=None, tf_list=None):
    """
    Apply biological network constraints to adjust predictions.

    This post-processing step uses biological knowledge to refine predictions:
    1. Non-TF sources should have lower probability (can't regulate)
    2. Highly variable targets are more likely to be regulated

    Args:
        predictions: Model predictions (probabilities) [num_edges]
        edges: Edge indices [2, num_edges]
        node_features: Node features [num_nodes, num_features]
        gene_names: List of gene names (optional)
        tf_list: List of TF gene names (optional)

    Returns:
        torch.Tensor: Adjusted predictions
    """
    predictions = predictions.clone() if torch.is_tensor(predictions) else predictions.copy()

    # Get TF indicators from node features (feature 3)
    if node_features is not None and node_features.shape[1] > 3:
        source_is_tf = node_features[edges[0], 3]

        # Reduce probability for non-TF sources
        # A non-TF gene is unlikely to regulate other genes
        non_tf_penalty = 0.3  # Multiply by 0.3 for non-TF sources
        tf_mask = source_is_tf == 1.0

        if torch.is_tensor(predictions):
            predictions = torch.where(tf_mask, predictions, predictions * non_tf_penalty)
        else:
            predictions = np.where(tf_mask.numpy(), predictions, predictions * non_tf_penalty)

    return predictions


def compute_metrics(predictions, labels, metric_list, threshold=0.5):
    """
    Compute evaluation metrics.

    - AUROC: Area under ROC curve
    - AUPRC: Area under PR curve
    - F1: Harmonic mean of precision and recall
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)

    Args:
        predictions: Model predictions (probabilities)
        labels: True labels
        metric_list: List of metrics to compute
        threshold: Classification threshold (default 0.5)

    Returns:
        dict: Computed metrics
    """
    # Convert to numpy
    pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    label_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    metrics = {}

    # Binary predictions using specified threshold
    pred_binary = (pred_np >= threshold).astype(int)

    if 'auroc' in metric_list:
        metrics['auroc'] = roc_auc_score(label_np, pred_np)

    if 'auprc' in metric_list:
        metrics['auprc'] = average_precision_score(label_np, pred_np)

    if 'f1' in metric_list:
        metrics['f1'] = f1_score(label_np, pred_binary)

    if 'precision' in metric_list:
        metrics['precision'] = precision_score(label_np, pred_binary)

    if 'recall' in metric_list:
        metrics['recall'] = recall_score(label_np, pred_binary)

    if 'accuracy' in metric_list:
        metrics['accuracy'] = accuracy_score(label_np, pred_binary)

    return metrics


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

