#!/bin/bash
# Example commands for GeneInference pipeline

echo "GeneInference Example Commands"
echo "=============================="
echo ""

# Function to show command
show_cmd() {
    echo ">>> $1"
    echo "    $2"
    echo ""
}

echo "1. QUICK START - Full Pipeline"
show_cmd "Run complete pipeline" \
        "python main.py --experiment-type full"

echo "2. STEP-BY-STEP PIPELINE"
show_cmd "Download data" \
        "python main.py --experiment-type download"
show_cmd "Preprocess data" \
        "python main.py --experiment-type preprocess"
show_cmd "Train model" \
        "python main.py --experiment-type train"
show_cmd "Evaluate model (replace EXPERIMENT_ID)" \
        "python main.py --experiment-type evaluate --eval-experiment-id EXPERIMENT_ID"
show_cmd "Predict GRN (replace EXPERIMENT_ID)" \
        "python main.py --experiment-type predict --trained-experiment-id EXPERIMENT_ID"

echo "3. EXPERIMENT MANAGEMENT"
show_cmd "List all experiments" \
        "python main.py --list-experiments"
show_cmd "Show best experiment" \
        "python main.py --best-experiment"
show_cmd "Show summary statistics" \
        "python main.py --summary"

echo "4. HYPERPARAMETER SWEEP"
show_cmd "Run sweep" \
        "python main.py --experiment-type sweep --sweep-config sweep_config.yaml"

echo "5. VISUALIZATION"
show_cmd "Generate overall visualizations dashboard" \
        "python visualize_results.py"
show_cmd "Copy best model visualizations to central folder" \
        "python copy_best_visualizations.py --top-n 5 --metric auroc"
show_cmd "View best visualizations (opens HTML page)" \
        "open visualizations/index.html"

echo "6. MIGRATION (one-time)"
show_cmd "Migrate old experiments (dry run)" \
        "python migrate_experiments.py --dry-run"
show_cmd "Actually migrate" \
        "python migrate_experiments.py"
show_cmd "Clean up old experiments (keep best 10)" \
        "python migrate_experiments.py --cleanup --keep-best 10"

echo "7. CUSTOM CONFIGURATIONS"
show_cmd "Train with custom config" \
        "python main.py --experiment-type train --config custom_config.yaml"
show_cmd "Add notes to experiment" \
        "python main.py --experiment-type train --notes 'Testing new feature'"

echo "8. RESUME TRAINING"
show_cmd "Resume from checkpoint (replace EXPERIMENT_ID)" \
        "python main.py --experiment-type train --resume-from EXPERIMENT_ID"

echo ""
echo "For more information:"
echo "  - Read README.md for full documentation"
echo "  - Read QUICK_START.md for getting started"
echo "  - Read RESTRUCTURING_SUMMARY.md for migration guide"
echo "  - Run 'python main.py --help' for all options"
