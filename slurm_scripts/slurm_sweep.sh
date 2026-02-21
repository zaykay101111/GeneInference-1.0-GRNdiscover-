#!/bin/bash
#SBATCH --job-name=geneinference_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=requeue
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j_sweep.out
#SBATCH --error=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j_sweep.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjz6f3@umsystem.edu

module purge
module load miniconda3
source activate geneinference

cd /mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-
mkdir -p logs

nvidia-smi

python3 main.py --experiment-type sweep --sweep-config sweep_config.yaml
