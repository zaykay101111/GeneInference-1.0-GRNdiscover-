"""
Experiment Management System for GeneInference Pipeline
Handles experiment tracking, storage, and retrieval
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class ExperimentManager:
    """Manages experiment lifecycle, storage, and tracking"""

    def __init__(self, base_dir: str = "./experiments", registry_path: str = "./experiment_registry.csv"):
        """
        Args:
            base_dir: Root directory for all experiments
            registry_path: Path to experiment registry CSV
        """
        self.base_dir = Path(base_dir)
        self.registry_path = Path(registry_path)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load registry
        self._init_registry()

    def _init_registry(self):
        """Initialize experiment registry if it doesn't exist"""
        if not self.registry_path.exists():
            # Create empty registry with schema
            columns = [
                'experiment_id', 'experiment_type', 'dataset', 'model_type',
                'activation', 'hidden_dim', 'num_layers', 'num_heads',
                'created_at', 'status', 'auroc', 'auprc', 'precision',
                'recall', 'f1', 'threshold', 'checkpoint_path',
                'edge_splits_path', 'results_path', 'config_path', 'notes'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.registry_path, index=False)

        self.registry = pd.read_csv(self.registry_path)

    def create_experiment(self,
                         experiment_type: str,
                         config: Dict[str, Any],
                         notes: str = "") -> str:
        """
        Create a new experiment with organized directory structure

        Args:
            experiment_type: Type of experiment (train, evaluate, sweep, etc.)
            config: Configuration dictionary
            notes: Optional notes about the experiment

        Returns:
            experiment_id: Unique identifier for this experiment
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config.get('model', {}).get('model_type', 'unknown')
        dataset = config.get('data', {}).get('dataset_name', 'unknown')
        activation = config.get('model', {}).get('activation', 'default')
        hidden_dim = config.get('model', {}).get('hidden_dim', 64)
        num_layers = config.get('model', {}).get('num_layers', 2)

        experiment_id = f"{timestamp}_{experiment_type}_{model_type}_{activation}_h{hidden_dim}_l{num_layers}_{dataset}"

        # Create directory structure
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        subdirs = ['checkpoints', 'edge_splits', 'results', 'predictions', 'logs']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)

        # Save config
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Register experiment
        new_entry = {
            'experiment_id': experiment_id,
            'experiment_type': experiment_type,
            'dataset': dataset,
            'model_type': model_type,
            'activation': activation,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': config.get('model', {}).get('num_heads', None),
            'created_at': timestamp,
            'status': 'created',
            'auroc': None,
            'auprc': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'threshold': None,
            'checkpoint_path': str(exp_dir / 'checkpoints'),
            'edge_splits_path': str(exp_dir / 'edge_splits'),
            'results_path': str(exp_dir / 'results'),
            'config_path': str(config_path),
            'notes': notes
        }

        # Append to registry
        self.registry = pd.concat([self.registry, pd.DataFrame([new_entry])], ignore_index=True)
        self.registry.to_csv(self.registry_path, index=False)

        print(f"✓ Created experiment: {experiment_id}")
        print(f"  Directory: {exp_dir}")

        return experiment_id

    def update_experiment(self,
                         experiment_id: str,
                         status: Optional[str] = None,
                         metrics: Optional[Dict[str, float]] = None,
                         **kwargs):
        """
        Update experiment status and metrics

        Args:
            experiment_id: Experiment to update
            status: New status (running, completed, failed)
            metrics: Dictionary of metric_name: value
            **kwargs: Additional fields to update
        """
        # Find experiment in registry
        idx = self.registry[self.registry['experiment_id'] == experiment_id].index

        if len(idx) == 0:
            raise ValueError(f"Experiment {experiment_id} not found in registry")

        idx = idx[0]

        # Update status
        if status:
            self.registry.at[idx, 'status'] = status

        # Update metrics
        if metrics:
            for metric_name, value in metrics.items():
                if metric_name in self.registry.columns:
                    self.registry.at[idx, metric_name] = value

        # Update additional fields
        for key, value in kwargs.items():
            if key in self.registry.columns:
                self.registry.at[idx, key] = value

        # Save registry
        self.registry.to_csv(self.registry_path, index=False)

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Get the directory path for an experiment"""
        return self.base_dir / experiment_id

    def get_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment configuration"""
        config_path = self.get_experiment_dir(experiment_id) / 'config.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def list_experiments(self,
                        experiment_type: Optional[str] = None,
                        dataset: Optional[str] = None,
                        model_type: Optional[str] = None,
                        status: Optional[str] = None) -> pd.DataFrame:
        """
        Query experiments by filters

        Args:
            experiment_type: Filter by type
            dataset: Filter by dataset
            model_type: Filter by model
            status: Filter by status

        Returns:
            Filtered DataFrame of experiments
        """
        df = self.registry.copy()

        if experiment_type:
            df = df[df['experiment_type'] == experiment_type]
        if dataset:
            df = df[df['dataset'] == dataset]
        if model_type:
            df = df[df['model_type'] == model_type]
        if status:
            df = df[df['status'] == status]

        return df

    def get_best_experiment(self,
                           metric: str = 'auroc',
                           dataset: Optional[str] = None,
                           model_type: Optional[str] = None) -> Optional[str]:
        """
        Find best experiment by metric

        Args:
            metric: Metric to optimize (auroc, auprc, f1, etc.)
            dataset: Filter by dataset
            model_type: Filter by model type

        Returns:
            experiment_id of best experiment or None
        """
        df = self.list_experiments(dataset=dataset, model_type=model_type, status='completed')

        if len(df) == 0 or metric not in df.columns:
            return None

        # Remove NaN values
        df = df.dropna(subset=[metric])

        if len(df) == 0:
            return None

        # Find best
        best_idx = df[metric].idxmax()
        return df.loc[best_idx, 'experiment_id']

    def save_results(self,
                    experiment_id: str,
                    results: Dict[str, Any],
                    filename: str = 'results.yaml'):
        """
        Save experiment results

        Args:
            experiment_id: Experiment ID
            results: Results dictionary
            filename: Filename for results
        """
        exp_dir = self.get_experiment_dir(experiment_id)
        results_path = exp_dir / 'results' / filename

        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        print(f"✓ Saved results to {results_path}")

    def save_checkpoint(self,
                       experiment_id: str,
                       checkpoint_data: Dict[str, Any],
                       filename: str = 'best_model.pt'):
        """
        Save model checkpoint

        Args:
            experiment_id: Experiment ID
            checkpoint_data: Checkpoint dictionary
            filename: Checkpoint filename
        """
        import torch

        exp_dir = self.get_experiment_dir(experiment_id)
        checkpoint_path = exp_dir / 'checkpoints' / filename

        torch.save(checkpoint_data, checkpoint_path)
        print(f"✓ Saved checkpoint to {checkpoint_path}")

        return str(checkpoint_path)

    def save_edge_splits(self,
                        experiment_id: str,
                        edge_splits: Dict[str, Any],
                        filename: str = 'edge_splits.pkl'):
        """
        Save edge splits

        Args:
            experiment_id: Experiment ID
            edge_splits: Edge split dictionary
            filename: Filename
        """
        import pickle

        exp_dir = self.get_experiment_dir(experiment_id)
        splits_path = exp_dir / 'edge_splits' / filename

        with open(splits_path, 'wb') as f:
            pickle.dump(edge_splits, f)

        print(f"✓ Saved edge splits to {splits_path}")

        return str(splits_path)

    def load_checkpoint(self, experiment_id: str, filename: str = 'best_model.pt'):
        """Load model checkpoint"""
        import torch

        exp_dir = self.get_experiment_dir(experiment_id)
        checkpoint_path = exp_dir / 'checkpoints' / filename

        return torch.load(checkpoint_path, weights_only=False)

    def load_edge_splits(self, experiment_id: str, filename: str = 'edge_splits.pkl'):
        """Load edge splits"""
        import pickle

        exp_dir = self.get_experiment_dir(experiment_id)
        splits_path = exp_dir / 'edge_splits' / filename

        with open(splits_path, 'rb') as f:
            return pickle.load(f)

    def export_summary(self, output_path: str = './experiment_summary.csv'):
        """Export experiment summary with key metrics"""
        summary_df = self.registry[['experiment_id', 'experiment_type', 'dataset',
                                    'model_type', 'activation', 'hidden_dim', 'num_layers',
                                    'auroc', 'auprc', 'precision', 'recall', 'f1',
                                    'threshold', 'status', 'created_at']].copy()

        summary_df.to_csv(output_path, index=False)
        print(f"✓ Exported summary to {output_path}")

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments side-by-side

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            DataFrame with comparison
        """
        df = self.registry[self.registry['experiment_id'].isin(experiment_ids)]
        return df[['experiment_id', 'model_type', 'activation', 'dataset',
                  'auroc', 'auprc', 'precision', 'recall', 'f1', 'threshold']]

    def cleanup_failed_experiments(self):
        """Remove failed experiments from disk and registry"""
        failed = self.registry[self.registry['status'] == 'failed']

        for _, row in failed.iterrows():
            exp_id = row['experiment_id']
            exp_dir = self.get_experiment_dir(exp_id)

            if exp_dir.exists():
                shutil.rmtree(exp_dir)
                print(f"✓ Removed failed experiment: {exp_id}")

        # Update registry
        self.registry = self.registry[self.registry['status'] != 'failed']
        self.registry.to_csv(self.registry_path, index=False)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all experiments"""
        stats = {
            'total_experiments': len(self.registry),
            'by_type': self.registry['experiment_type'].value_counts().to_dict(),
            'by_dataset': self.registry['dataset'].value_counts().to_dict(),
            'by_model': self.registry['model_type'].value_counts().to_dict(),
            'by_status': self.registry['status'].value_counts().to_dict(),
        }

        # Best metrics across all completed experiments
        completed = self.registry[self.registry['status'] == 'completed']
        if len(completed) > 0:
            stats['best_auroc'] = completed['auroc'].max()
            stats['best_auprc'] = completed['auprc'].max()
            stats['best_f1'] = completed['f1'].max()
            stats['avg_auroc'] = completed['auroc'].mean()
            stats['avg_auprc'] = completed['auprc'].mean()

        return stats
