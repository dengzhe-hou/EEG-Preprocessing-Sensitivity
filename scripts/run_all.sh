#!/bin/bash
# Run the full PGI experiment pipeline via Slurm.
# Usage: bash scripts/run_all.sh

set -e
cd /home/hou/Research/Unified-EEG-Preprocessing
mkdir -p logs

echo "=== Step 1: Data Preparation ==="
PREP_JOB=$(sbatch --parsable scripts/slurm_prepare.sh)
echo "Submitted data prep job: $PREP_JOB"

echo "=== Step 2: Fingerprinting (after data prep) ==="
FP_JOB=$(sbatch --parsable --dependency=afterok:$PREP_JOB scripts/slurm_fingerprint.sh)
echo "Submitted fingerprint job: $FP_JOB (depends on $PREP_JOB)"

echo "=== Step 3: Training experiments (after data prep) ==="
for METHOD in pgi erm_single erm_mixed consistency; do
    for MODEL in eegnet shallow; do
        for SEED in 42 43 44; do
            JOB=$(sbatch --parsable --dependency=afterok:$PREP_JOB \
                  scripts/slurm_train.sh $METHOD $MODEL $SEED)
            echo "Submitted: $METHOD/$MODEL/s$SEED → job $JOB"
        done
    done
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: squeue -u \$USER"
echo "Logs:    ls logs/"
