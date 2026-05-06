#!/bin/bash
#SBATCH --job-name=pgi-fp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/fingerprint_%j.log

cd /home/hou/Research/Unified-EEG-Preprocessing

eval "$(conda shell.bash hook)"
conda activate pgi

echo "=== Running pipeline fingerprinting experiments ==="
python scripts/fingerprint.py \
    --data-dir data/processed/bci_iv_2a \
    --output-dir results/fingerprint

echo "=== Done ==="
