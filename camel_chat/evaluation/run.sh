#!/bin/bash -l
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=128G
#SBATCH --reservation=A100

echo "Loading anaconda..."

conda activate lmeval

echo "...Anaconda env loaded"

python main.py --device cuda --no_cache "$@"

echo "[TRAIN] Done"