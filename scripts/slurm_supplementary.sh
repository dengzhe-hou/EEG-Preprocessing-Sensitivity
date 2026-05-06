#!/bin/bash
#SBATCH --job-name=supp_exp
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=c03
#SBATCH --exclude=c04
#SBATCH --output=logs/supp_%A_%a.log
#SBATCH --error=logs/supp_%A_%a.err

# Usage: sbatch --array=0-8 scripts/slurm_supplementary.sh
# Jobs 0-5: NA-PGI seeds 45,46 x 3 datasets (M1)
# Jobs 6-8: ShallowNet ERM x 3 datasets (M3)

eval "$(/home/hou/miniconda3/bin/conda shell.bash hook)"
conda activate pgi

cd /home/hou/Research/Unified-EEG-Preprocessing

TASK_ID=${SLURM_ARRAY_TASK_ID}

case $TASK_ID in
  # M1: NA-PGI multi-seed
  0) DATA=bci_iv_2a     SEED=45 METHOD=pgi  MODEL=eegnet OUTDIR=results_v4/napgi_bci_s45 ;;
  1) DATA=bci_iv_2a     SEED=46 METHOD=pgi  MODEL=eegnet OUTDIR=results_v4/napgi_bci_s46 ;;
  2) DATA=physionet_mi  SEED=45 METHOD=pgi  MODEL=eegnet OUTDIR=results_v4/napgi_physionet_s45 ;;
  3) DATA=physionet_mi  SEED=46 METHOD=pgi  MODEL=eegnet OUTDIR=results_v4/napgi_physionet_s46 ;;
  4) DATA=seed_iv       SEED=45 METHOD=pgi  MODEL=eegnet OUTDIR=results_v4/napgi_seed_iv_s45 ;;
  5) DATA=seed_iv       SEED=46 METHOD=pgi  MODEL=eegnet OUTDIR=results_v4/napgi_seed_iv_s46 ;;
  # M3: ShallowNet ERM
  6) DATA=bci_iv_2a     SEED=42 METHOD=erm_single MODEL=shallow OUTDIR=results_v4/shallow_erm_bci ;;
  7) DATA=physionet_mi  SEED=42 METHOD=erm_single MODEL=shallow OUTDIR=results_v4/shallow_erm_physionet ;;
  8) DATA=seed_iv       SEED=42 METHOD=erm_single MODEL=shallow OUTDIR=results_v4/shallow_erm_seed_iv ;;
esac

mkdir -p $OUTDIR

EXTRA_ARGS=""
if [ "$METHOD" = "pgi" ]; then
  EXTRA_ARGS="--normalize-pgi --adaptive-lambda --cfr-target 0.15 --n-edge-sample 256 --n-sup-views 8"
fi

echo "Running: method=$METHOD model=$MODEL data=$DATA seed=$SEED outdir=$OUTDIR"
echo "Extra args: $EXTRA_ARGS"

python scripts/train_pgi.py \
  --data-dir data/processed_v4/$DATA \
  --output-dir $OUTDIR \
  --method $METHOD \
  --model $MODEL \
  --seed $SEED \
  --n-folds 3 \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-3 \
  $EXTRA_ARGS

echo "Done: $OUTDIR"
