#!/bin/bash
#SBATCH --job-name=p300-pipe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --exclude=c04
#SBATCH --output=logs/p300_pipeline_%j.log

cd /home/hou/Research/Unified-EEG-Preprocessing
eval "$(conda shell.bash hook)"
conda activate pgi

echo "========================================="
echo "STEP 1: Data Preparation (128 pipelines)"
echo "========================================="
python scripts/prepare_p300_v4.py --output-dir data/processed_v4
echo ""

echo "========================================="
echo "STEP 2: ERM-single Training"
echo "========================================="
python scripts/train_pgi.py \
  --data-dir data/processed_v4/p300 \
  --output-dir results_v4/p300_erm_single \
  --method erm_single --model eegnet --epochs 50 \
  --n-pipelines 128 --n-views-per-batch 1 --batch-size 64 \
  --n-folds 3 --seed 42
echo ""

echo "========================================="
echo "STEP 3: Per-Intervention Analysis"
echo "========================================="
python scripts/analyze_interventions.py \
  --data-dir data/processed_v4/p300 \
  --model-dir results_v4/p300_erm_single \
  --output-dir results_v4/analysis_p300
echo ""

echo "========================================="
echo "ALL DONE"
echo "========================================="
