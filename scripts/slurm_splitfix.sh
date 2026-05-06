#!/bin/bash
#SBATCH --job-name=splitfix
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=c03
#SBATCH --output=logs/splitfix_%A_%a.log
#SBATCH --error=logs/splitfix_%A_%a.err

# Re-run experiments affected by subject split fix
# Jobs 0-2: ERM-single for PhysionetMI, P300, Lee2019 (sensitivity analysis)
# Jobs 3-4: DG baselines + NA-PGI for PhysionetMI (Tab:main)
# Jobs 5-8: NA-PGI PhysionetMI seeds 43-46
# Job 9: ShallowNet PhysionetMI

eval "$(/home/hou/miniconda3/bin/conda shell.bash hook)"
conda activate pgi
cd /home/hou/Research/Unified-EEG-Preprocessing

TASK_ID=${SLURM_ARRAY_TASK_ID}

case $TASK_ID in
  # ERM-single (sensitivity analysis baseline)
  0) DATA=physionet_mi  METHOD=erm_single MODEL=eegnet SEED=42 OUTDIR=results_v4/physionet_erm_single_v2 EXTRA="" ;;
  1) DATA=p300          METHOD=erm_single MODEL=eegnet SEED=42 OUTDIR=results_v4/p300_erm_single_v2 EXTRA="" ;;
  2) DATA=lee2019_erp   METHOD=erm_single MODEL=eegnet SEED=42 OUTDIR=results_v4/lee2019_erp_erm_single_v2 EXTRA="" ;;
  # Tab:main baselines for PhysionetMI
  3) DATA=physionet_mi  METHOD=erm_mixed  MODEL=eegnet SEED=42 OUTDIR=results_v4/erm_mixed_physionet_v2 EXTRA="" ;;
  4) DATA=physionet_mi  METHOD=consistency MODEL=eegnet SEED=42 OUTDIR=results_v4/consistency_physionet_v2 EXTRA="" ;;
  # NA-PGI PhysionetMI (all 5 seeds)
  5) DATA=physionet_mi  METHOD=pgi MODEL=eegnet SEED=42 OUTDIR=results_v4/napgi_physionet_v2 EXTRA="--normalize-pgi --adaptive-lambda --cfr-target 0.15 --n-edge-sample 256 --n-sup-views 8" ;;
  6) DATA=physionet_mi  METHOD=pgi MODEL=eegnet SEED=43 OUTDIR=results_v4/napgi_physionet_v2_s43 EXTRA="--normalize-pgi --adaptive-lambda --cfr-target 0.15 --n-edge-sample 256 --n-sup-views 8" ;;
  7) DATA=physionet_mi  METHOD=pgi MODEL=eegnet SEED=44 OUTDIR=results_v4/napgi_physionet_v2_s44 EXTRA="--normalize-pgi --adaptive-lambda --cfr-target 0.15 --n-edge-sample 256 --n-sup-views 8" ;;
  8) DATA=physionet_mi  METHOD=pgi MODEL=eegnet SEED=45 OUTDIR=results_v4/napgi_physionet_v2_s45 EXTRA="--normalize-pgi --adaptive-lambda --cfr-target 0.15 --n-edge-sample 256 --n-sup-views 8" ;;
  9) DATA=physionet_mi  METHOD=pgi MODEL=eegnet SEED=46 OUTDIR=results_v4/napgi_physionet_v2_s46 EXTRA="--normalize-pgi --adaptive-lambda --cfr-target 0.15 --n-edge-sample 256 --n-sup-views 8" ;;
  # ShallowNet PhysionetMI
  10) DATA=physionet_mi METHOD=erm_single MODEL=shallow SEED=42 OUTDIR=results_v4/shallow_erm_physionet_v2 EXTRA="" ;;
  # DG baselines for PhysionetMI
  11) DATA=physionet_mi METHOD=groupdro   MODEL=eegnet SEED=42 OUTDIR=results_v4/groupdro_physionet_v2 EXTRA="" ;;
  12) DATA=physionet_mi METHOD=irm        MODEL=eegnet SEED=42 OUTDIR=results_v4/irm_physionet_v2 EXTRA="" ;;
  13) DATA=physionet_mi METHOD=coral      MODEL=eegnet SEED=42 OUTDIR=results_v4/coral_physionet_v2 EXTRA="" ;;
esac

mkdir -p $OUTDIR

echo "Running: method=$METHOD model=$MODEL data=$DATA seed=$SEED outdir=$OUTDIR"
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
  $EXTRA

echo "Done: $OUTDIR"
