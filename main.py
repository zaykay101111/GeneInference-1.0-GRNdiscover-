"""
GeneInference: Unified Pipeline for Gene Regulatory Network Inference
Single entry point for all experiments

Usage:
    python main.py --experiment-type <type> [options]

Experiment Types:
    download    - Download dataset
    preprocess  - Preprocess data into graph format
    train       - Train a single model
    evaluate    - Evaluate a trained model
    predict     - Generate GRN predictions
    sweep       - Hyperparameter sweep
    ensemble    - Train ensemble models
    full        - Complete pipeline (download -> predict)
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.experiment_manager import ExperimentManager


def load_config(config_path: str = 'config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_experiment(config, experiment_id, exp_manager):
    """Run download experiment — downloads and sets up all datasets"""
    import os
    from src.data import download

    print("\n" + "="*80)
    print("DOWNLOAD EXPERIMENT")
    print("="*80)

    raw_data_dir = config['data']['raw_data_dir']

    all_datasets = (
        config['data']['datasets']['train'] +
        config['data']['datasets']['test']
    )
    print(f"\nDatasets: {len(all_datasets)}")
    print(f"  Train: {config['data']['datasets']['train']}")
    print(f"  Test:  {config['data']['datasets']['test']}")
    print(f"Output directory: {raw_data_dir}\n")

    try:
        exp_manager.update_experiment(experiment_id, status='running')

        # Download and set up all datasets
        download.download_dataset(config, raw_data_dir)

        # Verify all datasets
        results = download.verify_all_datasets(config, raw_data_dir)
        all_valid = all(results.values())

        if all_valid:
            exp_manager.update_experiment(experiment_id, status='completed')
            print(f"\n  Download completed: all {len(results)} datasets valid")
        else:
            failed = [k for k, v in results.items() if not v]
            exp_manager.update_experiment(experiment_id, status='failed')
            print(f"\n  Download verification failed for: {', '.join(failed)}")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n  Download failed: {e}")
        raise


def preprocess_experiment(config, experiment_id, exp_manager):
    """Run preprocessing experiment — processes all datasets"""
    from src.data import preprocess

    print("\n" + "="*80)
    print("PREPROCESSING EXPERIMENT")
    print("="*80)

    try:
        exp_manager.update_experiment(experiment_id, status='running')

        # Run multi-dataset preprocessing pipeline
        results = preprocess.preprocess_pipeline(config)

        # Save metadata
        exp_dir = exp_manager.get_experiment_dir(experiment_id)
        preprocess.save_metadata(config, str(exp_dir / 'results'))

        exp_manager.update_experiment(experiment_id, status='completed')

        print(f"\n  Preprocessing completed: {len(results)} datasets")
        for cell_type, (data, gene_names) in results.items():
            print(f"  [{cell_type}] {data.num_nodes} genes, "
                  f"{data.num_edges} co-expr edges, "
                  f"{data.y.shape[1]} GT edges")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n  Preprocessing failed: {e}")
        raise


def train_experiment(config, experiment_id, exp_manager, resume_from=None):
    """Run training experiment"""
    from src.training import trainer

    print("\n" + "="*80)
    print("TRAINING EXPERIMENT")
    print("="*80)

    # Get experiment directory
    exp_dir = exp_manager.get_experiment_dir(experiment_id)

    # Update config with experiment paths
    config['output']['checkpoint_dir'] = str(exp_dir / 'checkpoints')
    config['output']['log_dir'] = str(exp_dir / 'logs')

    # Set edge splits path for dataset module to use
    import os
    os.makedirs(str(exp_dir / 'edge_splits'), exist_ok=True)
    config['_edge_splits_path'] = str(exp_dir / 'edge_splits' / 'edge_splits.pkl')

    try:
        exp_manager.update_experiment(experiment_id, status='running')

        # Train model using existing train function
        print("\nStarting training...")
        trainer.train(config=config, quiet=True)

        # Load results from saved file
        import yaml
        results_path = os.path.join(config['output']['results_dir'], 'test_results.yaml')

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = yaml.safe_load(f)

            # Save results to experiment manager
            metrics = {
                'auroc': results['test_metrics'].get('auroc'),
                'auprc': results['test_metrics'].get('auprc'),
                'precision': results['test_metrics'].get('precision'),
                'recall': results['test_metrics'].get('recall'),
                'f1': results['test_metrics'].get('f1'),
                'threshold': results.get('optimal_threshold')
            }

            exp_manager.update_experiment(experiment_id, status='completed', metrics=metrics)
            exp_manager.save_results(experiment_id, results)

            print(f"\n✓ Training completed successfully")
            print(f"\nTest Metrics:")
            print(f"  AUROC:     {results['test_metrics']['auroc']:.4f}")
            print(f"  AUPRC:     {results['test_metrics']['auprc']:.4f}")
            print(f"  Precision: {results['test_metrics']['precision']:.4f}")
            print(f"  Recall:    {results['test_metrics']['recall']:.4f}")
            print(f"  F1:        {results['test_metrics']['f1']:.4f}")
            print(f"  Threshold: {results.get('optimal_threshold', 0.5):.4f}")

            # Show per-dataset results if available
            if 'per_dataset_metrics' in results:
                print(f"\n  Per-dataset test metrics:")
                for ct, m in results['per_dataset_metrics'].items():
                    print(f"    [{ct}] AUROC={m.get('auroc', 0):.4f} "
                          f"AUPRC={m.get('auprc', 0):.4f} F1={m.get('f1', 0):.4f}")
        else:
            exp_manager.update_experiment(experiment_id, status='completed')
            print(f"\n✓ Training completed")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n✗ Training failed: {e}")
        raise


def evaluate_experiment(config, experiment_id, exp_manager, eval_experiment_id=None):
    """Run evaluation experiment (supports multi-dataset)"""
    from src.evaluation import evaluator

    print("\n" + "="*80)
    print("EVALUATION EXPERIMENT")
    print("="*80)

    # If eval_experiment_id specified, evaluate that experiment
    if eval_experiment_id:
        eval_dir = exp_manager.get_experiment_dir(eval_experiment_id)
        checkpoint_path = eval_dir / 'checkpoints' / 'best_model.pt'

        # Load config from the experiment being evaluated
        eval_config_path = eval_dir / 'config.yaml'
        if eval_config_path.exists():
            import yaml
            with open(eval_config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"\n✓ Loaded config from experiment")

        print(f"\nEvaluating experiment: {eval_experiment_id}")
    else:
        print("\n✗ Must specify --eval-experiment-id")
        return

    # Get output directory for evaluation results
    exp_dir = exp_manager.get_experiment_dir(experiment_id)
    results_dir = exp_dir / 'results'

    try:
        exp_manager.update_experiment(experiment_id, status='running')

        # Run evaluation (handles multi-dataset internally)
        print("\nRunning evaluation...")
        results = evaluator.run_evaluation(
            config=config,
            checkpoint_path=str(checkpoint_path),
            output_dir=str(results_dir)
        )

        # Update experiment
        metrics = {
            'auroc': results['metrics']['auroc'],
            'auprc': results['metrics']['auprc'],
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1': results['metrics']['f1'],
        }

        exp_manager.update_experiment(experiment_id, status='completed', metrics=metrics)
        exp_manager.save_results(experiment_id, results)

        print(f"\n✓ Evaluation completed successfully")
        print(f"\nMetrics:")
        print(f"  AUROC:     {results['metrics']['auroc']:.4f}")
        print(f"  AUPRC:     {results['metrics']['auprc']:.4f}")
        print(f"  Precision: {results['metrics']['precision']:.4f}")
        print(f"  Recall:    {results['metrics']['recall']:.4f}")
        print(f"  F1:        {results['metrics']['f1']:.4f}")

        # Show per-dataset results if available
        if 'per_dataset' in results:
            print(f"\n  Per-dataset:")
            for ct, m in results['per_dataset'].items():
                print(f"    [{ct}] AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} F1={m['f1']:.4f}")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n✗ Evaluation failed: {e}")
        raise


def predict_experiment(config, experiment_id, exp_manager, trained_experiment_id=None):
    """Run prediction experiment to generate GRN (supports multi-dataset)"""
    from src.evaluation import predictor
    from src.data import dataset
    import torch

    print("\n" + "="*80)
    print("PREDICTION EXPERIMENT")
    print("="*80)

    # Get trained model
    if trained_experiment_id:
        trained_dir = exp_manager.get_experiment_dir(trained_experiment_id)
        checkpoint_path = trained_dir / 'checkpoints' / 'best_model.pt'

        # Load config from the trained experiment
        import yaml
        trained_config_path = trained_dir / 'config.yaml'
        if trained_config_path.exists():
            with open(trained_config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"\n✓ Loaded config from trained experiment")

        print(f"\nUsing model from: {trained_experiment_id}")
    else:
        print("\n✗ Must specify --trained-experiment-id")
        return

    # Get output directory
    exp_dir = exp_manager.get_experiment_dir(experiment_id)
    predictions_dir = exp_dir / 'predictions'

    try:
        exp_manager.update_experiment(experiment_id, status='running')

        # Determine if multi-dataset
        use_multi = ('datasets' in config['data'] and
                     'test' in config['data']['datasets'] and
                     len(config['data']['datasets']['test']) > 0)

        # Load model
        from src.models import architectures
        from src.utils import helpers

        if use_multi:
            # Get num features from first training dataset
            first_train = config['data']['datasets']['train'][0].replace('beeline_', '')
            first_data, _ = dataset.load_single_dataset(config, first_train)
            num_features = first_data.x.shape[1]
        else:
            data, gene_names = dataset.load_preprocessed_data(config)
            num_features = data.x.shape[1]

        model = architectures.create_model(config, num_features)
        checkpoint = helpers.load_checkpoint(str(checkpoint_path), model)
        model.eval()
        threshold = checkpoint.get('optimal_threshold', 0.5)

        import os
        os.makedirs(predictions_dir, exist_ok=True)

        if use_multi:
            # Generate predictions for each test dataset
            test_dataset_names = config['data']['datasets']['test']
            for dataset_name in test_dataset_names:
                cell_type = dataset_name.replace('beeline_', '')
                print(f"\nGenerating predictions for {cell_type}...")

                ct_data, ct_gene_names = dataset.load_single_dataset(config, cell_type)

                predictions_df = predictor.predict_grn(
                    model=model,
                    data=ct_data,
                    gene_names=ct_gene_names,
                    threshold=threshold
                )

                output_path = predictions_dir / f'predicted_grn_{cell_type}.csv'
                predictions_df.to_csv(output_path, index=False)

                top_path = predictions_dir / f'top_100_{cell_type}.csv'
                predictions_df.head(100).to_csv(top_path, index=False)

                print(f"  [{cell_type}] {len(predictions_df)} predictions, "
                      f"{predictions_df['predicted_regulatory'].sum()} regulatory")
        else:
            # Single-dataset mode
            predictions_df = predictor.predict_grn(
                model=model, data=data, gene_names=gene_names, threshold=threshold
            )
            output_path = predictions_dir / 'predicted_grn.csv'
            predictions_df.to_csv(output_path, index=False)
            top_path = predictions_dir / 'top_100_predictions.csv'
            predictions_df.head(100).to_csv(top_path, index=False)

            print(f"\n  Total predictions: {len(predictions_df)}")
            print(f"  Predicted regulatory edges: {predictions_df['predicted_regulatory'].sum()}")

        exp_manager.update_experiment(experiment_id, status='completed')
        print(f"\n✓ Prediction completed successfully")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n✗ Prediction failed: {e}")
        raise


def sweep_experiment(config, experiment_id, exp_manager, sweep_config_path):
    """Run hyperparameter sweep experiment"""
    from src.training import trainer
    from src.data import dataset
    import itertools

    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP EXPERIMENT")
    print("="*80)

    # Load sweep configuration
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Generate all combinations
    param_names = list(sweep_config.keys())
    param_values = [sweep_config[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    print(f"\nTotal configurations to test: {len(combinations)}")

    try:
        exp_manager.update_experiment(experiment_id, status='running')

        # Load data once
        print("\nLoading preprocessed data...")
        data, gene_names = dataset.load_preprocessed_data(config)

        # Results storage
        all_results = []

        # Run each configuration
        for i, combination in enumerate(combinations, 1):
            # Create config for this run
            run_config = config.copy()
            for param_name, param_value in zip(param_names, combination):
                # Update nested config
                keys = param_name.split('.')
                cfg = run_config
                for key in keys[:-1]:
                    cfg = cfg[key]
                cfg[keys[-1]] = param_value

            # Create sub-experiment
            run_name = "_".join([f"{name.split('.')[-1]}={value}" for name, value in zip(param_names, combination)])
            print(f"\n[{i}/{len(combinations)}] Running: {run_name}")

            sub_exp_id = exp_manager.create_experiment(
                experiment_type='sweep_run',
                config=run_config,
                notes=f"Sweep run {i}/{len(combinations)}: {run_name}"
            )

            # Train
            results = train_experiment(run_config, sub_exp_id, exp_manager)

            # Record results
            result_entry = {
                'run_name': run_name,
                'experiment_id': sub_exp_id,
                **{name: value for name, value in zip(param_names, combination)},
            }

            # Only add metrics if training returned results
            if results and 'test_metrics' in results:
                result_entry.update(results['test_metrics'])
                
            all_results.append(result_entry)

        # Save sweep results
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        exp_dir = exp_manager.get_experiment_dir(experiment_id)
        results_df.to_csv(exp_dir / 'results' / 'sweep_results.csv', index=False)

        # Find best configuration
        best_idx = results_df['auroc'].idxmax()
        best_config = results_df.iloc[best_idx]

        exp_manager.update_experiment(experiment_id, status='completed')

        print(f"\n✓ Sweep completed successfully")
        print(f"\nBest configuration:")
        print(f"  {best_config['run_name']}")
        print(f"  AUROC: {best_config['auroc']:.4f}")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n✗ Sweep failed: {e}")
        raise


def full_pipeline(config, experiment_id, exp_manager):
    """Run full pipeline: download -> preprocess -> train -> evaluate -> predict"""
    print("\n" + "="*80)
    print("FULL PIPELINE EXPERIMENT")
    print("="*80)

    try:
        # Step 1: Download
        print("\n" + "-"*80)
        print("Step 1/5: Download")
        print("-"*80)
        download_exp_id = exp_manager.create_experiment(
            experiment_type='download',
            config=config,
            notes=f"Full pipeline step 1: Download"
        )
        download_experiment(config, download_exp_id, exp_manager)

        # Step 2: Preprocess
        print("\n" + "-"*80)
        print("Step 2/5: Preprocess")
        print("-"*80)
        preprocess_exp_id = exp_manager.create_experiment(
            experiment_type='preprocess',
            config=config,
            notes=f"Full pipeline step 2: Preprocess"
        )
        preprocess_experiment(config, preprocess_exp_id, exp_manager)

        # Step 3: Train
        print("\n" + "-"*80)
        print("Step 3/5: Train")
        print("-"*80)
        train_exp_id = exp_manager.create_experiment(
            experiment_type='train',
            config=config,
            notes=f"Full pipeline step 3: Train"
        )
        train_experiment(config, train_exp_id, exp_manager)

        # Step 4: Evaluate
        print("\n" + "-"*80)
        print("Step 4/5: Evaluate")
        print("-"*80)
        eval_exp_id = exp_manager.create_experiment(
            experiment_type='evaluate',
            config=config,
            notes=f"Full pipeline step 4: Evaluate"
        )
        evaluate_experiment(config, eval_exp_id, exp_manager, eval_experiment_id=train_exp_id)

        # Step 5: Predict
        print("\n" + "-"*80)
        print("Step 5/5: Predict")
        print("-"*80)
        predict_exp_id = exp_manager.create_experiment(
            experiment_type='predict',
            config=config,
            notes=f"Full pipeline step 5: Predict"
        )
        predict_experiment(config, predict_exp_id, exp_manager, trained_experiment_id=train_exp_id)

        exp_manager.update_experiment(experiment_id, status='completed')

        print("\n" + "="*80)
        print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nExperiment IDs:")
        print(f"  Download:   {download_exp_id}")
        print(f"  Preprocess: {preprocess_exp_id}")
        print(f"  Train:      {train_exp_id}")
        print(f"  Evaluate:   {eval_exp_id}")
        print(f"  Predict:    {predict_exp_id}")

    except Exception as e:
        exp_manager.update_experiment(experiment_id, status='failed')
        print(f"\n✗ Full pipeline failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='GeneInference: Unified Pipeline for Gene Regulatory Network Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments (but not when using utility commands)
    parser.add_argument('--experiment-type', type=str,
                       choices=['download', 'preprocess', 'train', 'evaluate', 'predict', 'sweep', 'ensemble', 'full'],
                       help='Type of experiment to run')

    # Optional arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--notes', type=str, default='',
                       help='Notes about this experiment')

    # Experiment-specific arguments
    parser.add_argument('--eval-experiment-id', type=str,
                       help='Experiment ID to evaluate (for evaluate experiment type)')
    parser.add_argument('--trained-experiment-id', type=str,
                       help='Trained experiment ID to use for prediction')
    parser.add_argument('--sweep-config', type=str,
                       help='Path to sweep configuration file')
    parser.add_argument('--resume-from', type=str,
                       help='Experiment ID to resume training from')
    parser.add_argument('--cell-type', type=str, default=None,
                       help='Optional: process only this cell type (e.g., hESC)')

    # Utility arguments
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all experiments and exit')
    parser.add_argument('--best-experiment', action='store_true',
                       help='Show best experiment and exit')
    parser.add_argument('--summary', action='store_true',
                       help='Show experiment summary statistics and exit')

    args = parser.parse_args()

    # Initialize experiment manager
    exp_manager = ExperimentManager()

    # Handle utility commands
    if args.list_experiments:
        print("\nAll Experiments:")
        print(exp_manager.registry.to_string())
        return

    if args.best_experiment:
        best_id = exp_manager.get_best_experiment(metric='auroc')
        if best_id:
            print(f"\nBest Experiment (by AUROC): {best_id}")
            best_exp = exp_manager.registry[exp_manager.registry['experiment_id'] == best_id].iloc[0]
            print(f"  AUROC: {best_exp['auroc']:.4f}")
            print(f"  AUPRC: {best_exp['auprc']:.4f}")
        else:
            print("\nNo completed experiments found")
        return

    if args.summary:
        stats = exp_manager.get_summary_stats()
        print("\nExperiment Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    # Check if experiment_type is provided
    if not args.experiment_type:
        parser.error("--experiment-type is required")

    # Load configuration
    config = load_config(args.config)

    # Create experiment
    experiment_id = exp_manager.create_experiment(
        experiment_type=args.experiment_type,
        config=config,
        notes=args.notes
    )

    # Run appropriate experiment type
    if args.experiment_type == 'download':
        download_experiment(config, experiment_id, exp_manager)

    elif args.experiment_type == 'preprocess':
        preprocess_experiment(config, experiment_id, exp_manager)

    elif args.experiment_type == 'train':
        train_experiment(config, experiment_id, exp_manager, resume_from=args.resume_from)

    elif args.experiment_type == 'evaluate':
        evaluate_experiment(config, experiment_id, exp_manager, eval_experiment_id=args.eval_experiment_id)

    elif args.experiment_type == 'predict':
        predict_experiment(config, experiment_id, exp_manager, trained_experiment_id=args.trained_experiment_id)

    elif args.experiment_type == 'sweep':
        if not args.sweep_config:
            print("✗ --sweep-config required for sweep experiment type")
            sys.exit(1)
        sweep_experiment(config, experiment_id, exp_manager, args.sweep_config)

    elif args.experiment_type == 'full':
        full_pipeline(config, experiment_id, exp_manager)

    # Print experiment summary
    print(f"\n" + "="*80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Type: {args.experiment_type}")
    print(f"Directory: {exp_manager.get_experiment_dir(experiment_id)}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
