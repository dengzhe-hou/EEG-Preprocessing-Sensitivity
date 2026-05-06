#!/bin/bash
#SBATCH --job-name=sleep-pipe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclude=c04
#SBATCH --output=logs/sleep_pipeline_%j.log

# Full pipeline: data prep → ERM → PGI → analysis for Sleep-EDF
cd /home/hou/Research/Unified-EEG-Preprocessing
eval "$(conda shell.bash hook)"
conda activate pgi

echo "========================================="
echo "STEP 1: Data Preparation (128 pipelines)"
echo "========================================="
python scripts/prepare_sleep_edf_v4.py --output-dir data/processed_v4 --n-subjects 15
echo ""

echo "========================================="
echo "STEP 2: ERM-single Training"
echo "========================================="
python scripts/train_pgi.py \
  --data-dir data/processed_v4/sleep_edf \
  --output-dir results_v4/sleep_erm_single \
  --method erm_single --model eegnet --epochs 50 \
  --n-pipelines 128 --n-views-per-batch 1 --batch-size 64 \
  --n-folds 3 --seed 42
echo ""

echo "========================================="
echo "STEP 3: PGI Training (128 edges, λ=1.0)"
echo "========================================="
python scripts/train_pgi.py \
  --data-dir data/processed_v4/sleep_edf \
  --output-dir results_v4/sleep_pgi \
  --method pgi --model eegnet --pgi-lambda 1.0 --epochs 50 \
  --n-pipelines 128 --n-edge-sample 128 --n-sup-views 16 --batch-size 16 \
  --n-folds 3 --seed 42
echo ""

echo "========================================="
echo "STEP 4: Per-Intervention Analysis"
echo "========================================="
python scripts/analyze_interventions.py \
  --data-dir data/processed_v4/sleep_edf \
  --model-dir results_v4/sleep_erm_single \
  --output-dir results_v4/analysis_sleep
echo ""

echo "========================================="
echo "ALL DONE"
echo "========================================="
