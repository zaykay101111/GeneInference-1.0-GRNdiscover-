#!/bin/bash
#SBATCH --job-name=geneinfer_custom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j.out
#SBATCH --error=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kjz6f3@umsystem.edu

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

cd /mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-

source ~/.bashrc
conda activate geneinference

python main.py --experiment-type train --config costom_config.yaml

echo "End: $(date)"
