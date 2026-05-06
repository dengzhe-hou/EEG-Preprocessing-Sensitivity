#!/bin/bash
#SBATCH --job-name=pgi-train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log

# Usage: sbatch slurm_train.sh <method> <model> <seed>
# Example: sbatch slurm_train.sh pgi eegnet 42

METHOD=${1:-pgi}
MODEL=${2:-eegnet}
SEED=${3:-42}

cd /home/hou/Research/Unified-EEG-Preprocessing

eval "$(conda shell.bash hook)"
conda activate pgi

echo "=== Training: method=$METHOD model=$MODEL seed=$SEED ==="
python scripts/train_pgi.py \
    --data-dir data/processed/bci_iv_2a \
    --output-dir results/${METHOD}_${MODEL}_s${SEED} \
    --method $METHOD \
    --model $MODEL \
    --seed $SEED \
    --batch-size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --pgi-lambda 1.0 \
    --n-views-per-batch 4

echo "=== Done ==="
