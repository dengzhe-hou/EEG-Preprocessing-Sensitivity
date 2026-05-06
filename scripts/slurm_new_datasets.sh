#!/bin/bash
#SBATCH --job-name=new-data
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=c04
#SBATCH --output=logs/new_datasets_%j.log

# Full pipeline for one new dataset: data prep → ERM → analysis → extensions
DATASET=${1:-physionet_mi}

cd /home/hou/Research/Unified-EEG-Preprocessing
eval "$(conda shell.bash hook)"
conda activate pgi

echo "============================================"
echo "DATASET: $DATASET"
echo "============================================"

if [ "$DATASET" = "physionet_mi" ]; then
    echo "=== STEP 1: Data Prep (PhysionetMI, 20 subjects, 64ch) ==="
    python scripts/prepare_physionet_mi_v4.py --output-dir data/processed_v4 --n-subjects 20

    echo "=== STEP 2: ERM Training ==="
    python scripts/train_pgi.py \
      --data-dir data/processed_v4/physionet_mi \
      --output-dir results_v4/physionet_erm_single \
      --method erm_single --model eegnet --epochs 50 \
      --n-pipelines 128 --n-views-per-batch 1 --batch-size 64 \
      --n-folds 3 --seed 42

    echo "=== STEP 3: Per-Intervention Analysis ==="
    python scripts/analyze_interventions.py \
      --data-dir data/processed_v4/physionet_mi \
      --model-dir results_v4/physionet_erm_single \
      --output-dir results_v4/analysis_physionet

    echo "=== STEP 4: Extensions (Walsh-Hadamard + Mediation + Uncertainty) ==="
    python scripts/analyze_extensions.py 2>&1 | grep -E "EXTENSION|Saved|==="

elif [ "$DATASET" = "lee2019_erp" ]; then
    echo "=== STEP 1: Data Prep (Lee2019_ERP, 20 subjects, 62ch) ==="
    python scripts/prepare_lee2019_erp_v4.py --output-dir data/processed_v4 --n-subjects 20

    echo "=== STEP 2: ERM Training ==="
    python scripts/train_pgi.py \
      --data-dir data/processed_v4/lee2019_erp \
      --output-dir results_v4/lee2019_erp_erm_single \
      --method erm_single --model eegnet --epochs 50 \
      --n-pipelines 128 --n-views-per-batch 1 --batch-size 64 \
      --n-folds 3 --seed 42

    echo "=== STEP 3: Per-Intervention Analysis ==="
    python scripts/analyze_interventions.py \
      --data-dir data/processed_v4/lee2019_erp \
      --model-dir results_v4/lee2019_erp_erm_single \
      --output-dir results_v4/analysis_lee2019_erp

elif [ "$DATASET" = "chbmit" ]; then
    echo "=== STEP 1: Data Prep (CHB-MIT, 10 subjects, seizure) ==="
    python scripts/prepare_chbmit_v4.py --output-dir data/processed_v4 --n-subjects 10

    echo "=== STEP 2: ERM Training ==="
    python scripts/train_pgi.py \
      --data-dir data/processed_v4/chbmit \
      --output-dir results_v4/chbmit_erm_single \
      --method erm_single --model eegnet --epochs 50 \
      --n-pipelines 128 --n-views-per-batch 1 --batch-size 64 \
      --n-folds 3 --seed 42

    echo "=== STEP 3: Per-Intervention Analysis ==="
    python scripts/analyze_interventions.py \
      --data-dir data/processed_v4/chbmit \
      --model-dir results_v4/chbmit_erm_single \
      --output-dir results_v4/analysis_chbmit
fi

echo "============================================"
echo "DONE: $DATASET"
echo "============================================"
