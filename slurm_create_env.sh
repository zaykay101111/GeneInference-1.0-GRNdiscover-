#!/bin/bash
#SBATCH --job-name=recreate_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j_env.out
#SBATCH --error=/mnt/pixstor/data/kjz6f3/GeneInference-1.0-GRNdiscover-/logs/%j_env.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjz6f3@umsystem.edu

module purge
module load miniconda3

conda create -n geneinference python=3.11 -y
source activate geneinference

pip install pyyaml numpy pandas scipy scikit-learn matplotlib seaborn tensorboard
pip install torch torch_geometric
