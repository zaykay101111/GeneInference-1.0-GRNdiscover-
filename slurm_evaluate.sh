#!/bin/bash
#SBATCH --job-name=geneinference_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=requeue
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j_evaluate.out
#SBATCH --error=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j_evaluate.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjz6f3@umsystem.edu

module purge
module load miniconda3
source activate geneinference

cd /mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-
mkdir -p logs

# Set the experiment ID to evaluate
EXPERIMENT_ID=EXPERIMENT_ID="20260219_191043_sweep_run_transformer_leaky_relu_h64_l2_beeline_hESC""

python3 main.py --experiment-type evaluate --eval-experiment-id $EXPERIMENT_ID

