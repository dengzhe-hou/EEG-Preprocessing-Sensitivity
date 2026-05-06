#!/bin/bash
#SBATCH --job-name=seediv_retry
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=c03
#SBATCH --exclude=c04
#SBATCH --output=logs/seediv_retry_%A_%a.log
#SBATCH --error=logs/seediv_retry_%A_%a.err

# Usage: sbatch --array=0-1 scripts/slurm_seediv_retry.sh

eval "$(/home/hou/miniconda3/bin/conda shell.bash hook)"
conda activate pgi
cd /home/hou/Research/Unified-EEG-Preprocessing

TASK_ID=${SLURM_ARRAY_TASK_ID}

case $TASK_ID in
  0) SEED=45 OUTDIR=results_v4/napgi_seed_iv_s45 ;;
  1) SEED=46 OUTDIR=results_v4/napgi_seed_iv_s46 ;;
esac

# Clean previous failed attempt
rm -f $OUTDIR/*.pt

echo "Running: NA-PGI SEED-IV seed=$SEED (batch=32 to avoid 32-bit overflow)"

python scripts/train_pgi.py \
  --data-dir data/processed_v4/seed_iv \
  --output-dir $OUTDIR \
  --method pgi \
  --model eegnet \
  --seed $SEED \
  --n-folds 3 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --normalize-pgi --adaptive-lambda --cfr-target 0.15 \
  --n-edge-sample 128 \
  --n-sup-views 8

echo "Done: $OUTDIR"
