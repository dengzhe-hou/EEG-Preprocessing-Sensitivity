#!/bin/bash
#SBATCH --job-name=pgi-r2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclude=c04
#SBATCH --output=logs/r2_%x_%j.log

# Generic round-2 experiment runner
# Usage: sbatch slurm_round2.sh <script> <args...>

SCRIPT=$1
shift

cd /home/hou/Research/Unified-EEG-Preprocessing
eval "$(conda shell.bash hook)"
conda activate pgi

echo "=== Running: python $SCRIPT $@ ==="
python "$SCRIPT" "$@"
echo "=== Done ==="
