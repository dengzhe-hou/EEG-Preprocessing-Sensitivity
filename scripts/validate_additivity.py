#!/usr/bin/env python
"""
Validate the additivity finding: step-by-step greedy optimization ≈ oracle.

Uses existing per_pipeline_accuracy.npy — NO GPU needed.
For each dataset:
  1. On dev folds, for each intervention independently, pick the better setting
  2. Compose the 7 greedy-best settings into one pipeline
  3. Compare: all-off default vs greedy vs oracle best-of-128
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import INTERVENTION_NAMES, K, N_PIPELINES

RDIR = Path('/home/hou/Research/Unified-EEG-Preprocessing/results_v4')

DATASETS = {
    'bci': 'analysis_bci',
    'sleep': 'analysis_sleep',
    'p300': 'analysis_p300',
    'physionet': 'analysis_physionet',
    'seed_iv': 'analysis_seed_iv',
    'lee2019_erp': 'analysis_lee2019_erp',
}

results = {}

print("="*70)
print("ADDITIVITY VALIDATION: Step-by-Step Greedy vs Oracle")
print("="*70)

for dname, analysis_dir in DATASETS.items():
    acc_file = RDIR / analysis_dir / 'per_pipeline_accuracy.npy'
    if not acc_file.exists():
        print(f"\n{dname}: no data, skipping")
        continue

    acc = np.load(acc_file)  # (n_folds, 128)
    n_folds = acc.shape[0]

    print(f"\n{'='*50}")
    print(f"Dataset: {dname} ({n_folds} folds, {N_PIPELINES} pipelines)")
    print(f"{'='*50}")

    greedy_accs = []
    oracle_accs = []
    default_accs = []

    for test_fold in range(n_folds):
        # Dev folds = all except test_fold
        dev_folds = [i for i in range(n_folds) if i != test_fold]
        dev_acc = acc[dev_folds].mean(axis=0)  # (128,) average accuracy on dev
        test_acc = acc[test_fold]  # (128,) accuracy on test

        # Default: pipeline 0 (all interventions off)
        default_accs.append(test_acc[0])

        # Oracle: best pipeline on TEST (cheating, upper bound)
        oracle_accs.append(test_acc.max())

        # Greedy: for each intervention, pick best setting on DEV
        greedy_bits = []
        for k in range(K):
            # Compare all pipelines with bit k=0 vs bit k=1
            mask0 = np.array([(p >> k) & 1 == 0 for p in range(N_PIPELINES)])
            mask1 = ~mask0
            acc0 = dev_acc[mask0].mean()
            acc1 = dev_acc[mask1].mean()
            greedy_bits.append(1 if acc1 > acc0 else 0)

        # Compose greedy pipeline
        greedy_idx = sum(b << k for k, b in enumerate(greedy_bits))
        greedy_accs.append(test_acc[greedy_idx])

        bits_str = ''.join(str(b) for b in greedy_bits)
        print(f"  Fold {test_fold}: default={test_acc[0]:.4f}, "
              f"greedy(idx={greedy_idx}, bits={bits_str})={test_acc[greedy_idx]:.4f}, "
              f"oracle={test_acc.max():.4f}")

    default_mean = np.mean(default_accs)
    greedy_mean = np.mean(greedy_accs)
    oracle_mean = np.mean(oracle_accs)
    greedy_oracle_gap = oracle_mean - greedy_mean

    print(f"\n  Summary:")
    print(f"    Default (all-off):  {default_mean:.4f}")
    print(f"    Greedy (step-wise): {greedy_mean:.4f} (+{greedy_mean-default_mean:.4f} vs default)")
    print(f"    Oracle (best-128):  {oracle_mean:.4f}")
    print(f"    Greedy-Oracle gap:  {greedy_oracle_gap:.4f} {'✓ SMALL' if abs(greedy_oracle_gap) < 0.02 else '✗ LARGE'}")

    results[dname] = {
        'default': float(default_mean),
        'greedy': float(greedy_mean),
        'oracle': float(oracle_mean),
        'greedy_oracle_gap': float(greedy_oracle_gap),
        'greedy_improvement': float(greedy_mean - default_mean),
    }

# Save
out_file = RDIR / 'extensions' / 'additivity_validation.json'
out_file.parent.mkdir(exist_ok=True)
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_file}")
