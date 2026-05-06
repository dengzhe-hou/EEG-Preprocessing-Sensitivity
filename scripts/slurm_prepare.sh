#!/bin/bash
#SBATCH --job-name=pgi-prep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_%j.log

# Data preparation (CPU only, no GPU needed)
cd /home/hou/Research/Unified-EEG-Preprocessing

eval "$(conda shell.bash hook)"
conda activate pgi

echo "=== Preparing BCI-IV-2a dataset ==="
python scripts/prepare_bci_iv_2a.py --output-dir data/processed

echo "=== Done ==="
