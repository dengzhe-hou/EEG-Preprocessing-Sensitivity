#!/bin/bash
#SBATCH --job-name=pgi-prep4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep_v4_%j.log

cd /home/hou/Research/Unified-EEG-Preprocessing
eval "$(conda shell.bash hook)"
conda activate pgi

DATASET=${1:-bci}

if [ "$DATASET" = "bci" ]; then
    echo "=== Preparing BCI-IV-2a (128 pipelines, v4) ==="
    python scripts/prepare_bci_v4.py --output-dir data/processed_v4
elif [ "$DATASET" = "sleep" ]; then
    echo "=== Preparing Sleep-EDF (128 pipelines, v4) ==="
    python scripts/prepare_sleep_edf.py --output-dir data/processed_v4 --n-subjects 20
fi

echo "=== Done ==="
