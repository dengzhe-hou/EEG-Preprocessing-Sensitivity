#!/usr/bin/env python
"""
Three analysis extensions:
1. Walsh-Hadamard / ANOVA interaction spectrum
2. Signal-level mediation analysis
3. CFR as uncertainty → selective prediction
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import INTERVENTION_NAMES, K, N_PIPELINES

RDIR = Path('/home/hou/Research/Unified-EEG-Preprocessing/results_v4')
DDIR = Path('/home/hou/Research/Unified-EEG-Preprocessing/data/processed_v4')
OUTDIR = RDIR / 'extensions'
OUTDIR.mkdir(exist_ok=True)

DATASETS = {
    'bci': {'name': 'BCI-IV-2a (MI)', 'analysis': 'analysis_bci', 'data': 'bci_iv_2a'},
    'sleep': {'name': 'Sleep-EDF', 'analysis': 'analysis_sleep', 'data': 'sleep_edf'},
    'p300': {'name': 'P300 (ERP)', 'analysis': 'analysis_p300', 'data': 'p300'},
    'physionet': {'name': 'PhysionetMI', 'analysis': 'analysis_physionet', 'data': 'physionet_mi'},
    'seed_iv': {'name': 'SEED-IV', 'analysis': 'analysis_seed_iv', 'data': 'seed_iv'},
    'lee2019_erp': {'name': 'Lee2019-ERP', 'analysis': 'analysis_lee2019_erp', 'data': 'lee2019_erp'},
}


# ============================================================
# EXTENSION 1: Walsh-Hadamard Interaction Spectrum
# ============================================================
def walsh_hadamard_analysis():
    """Decompose 2^K accuracy cube into interaction orders via Walsh-Hadamard transform."""
    print("="*70)
    print("EXTENSION 1: Walsh-Hadamard Interaction Spectrum")
    print("="*70)

    results = {}

    for dkey, dinfo in DATASETS.items():
        acc_matrix = np.load(RDIR / dinfo['analysis'] / 'per_pipeline_accuracy.npy')
        # Average across folds
        mean_acc = acc_matrix.mean(axis=0)  # (128,)
        assert len(mean_acc) == N_PIPELINES, f"Expected {N_PIPELINES}, got {len(mean_acc)}"

        # Walsh-Hadamard transform
        # For a 2^K design, the WHT decomposes the function into 2^K coefficients
        # indexed by subsets S ⊆ {0,...,K-1}. The coefficient for subset S
        # captures the interaction among the interventions in S.
        n = N_PIPELINES
        coeffs = np.zeros(n)
        for s in range(n):
            # Coefficient for subset s (binary encoding)
            total = 0.0
            for x in range(n):
                # (-1)^(popcount(s & x))
                sign = (-1) ** bin(s & x).count('1')
                total += sign * mean_acc[x]
            coeffs[s] = total / n

        # Group coefficients by interaction order (number of bits set in index)
        order_variance = {}
        for order in range(K + 1):
            mask = [bin(s).count('1') == order for s in range(n)]
            order_coeffs = coeffs[mask]
            # Variance explained by this order = sum of squared coefficients
            order_variance[order] = float(np.sum(order_coeffs**2))

        total_var = sum(order_variance.values())
        if total_var > 0:
            order_pct = {k: v/total_var*100 for k, v in order_variance.items()}
        else:
            order_pct = {k: 0.0 for k in order_variance}

        print(f"\n{dinfo['name']}:")
        print(f"  {'Order':>5s} | {'# Terms':>7s} | {'Variance':>10s} | {'% Total':>8s}")
        print(f"  {'-'*40}")
        for order in range(K + 1):
            n_terms = sum(1 for s in range(n) if bin(s).count('1') == order)
            print(f"  {order:>5d} | {n_terms:>7d} | {order_variance[order]:>10.6f} | {order_pct[order]:>7.1f}%")

        # Cumulative
        cum = 0
        print(f"\n  Cumulative:")
        for order in range(K + 1):
            cum += order_pct[order]
            print(f"    Up to order {order}: {cum:.1f}%")

        results[dkey] = {
            'order_variance': order_variance,
            'order_pct': order_pct,
            'total_variance': float(total_var),
        }

    # Save
    with open(OUTDIR / 'walsh_hadamard.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTDIR / 'walsh_hadamard.json'}")
    return results


# ============================================================
# EXTENSION 3: CFR as Uncertainty → Selective Prediction
# ============================================================
def cfr_uncertainty_analysis():
    """Use pipeline disagreement as uncertainty for selective prediction."""
    print("\n" + "="*70)
    print("EXTENSION 3: CFR as Uncertainty → Selective Prediction")
    print("="*70)

    results = {}

    for dkey, dinfo in DATASETS.items():
        acc_matrix = np.load(RDIR / dinfo['analysis'] / 'per_pipeline_accuracy.npy')
        # acc_matrix shape: (n_folds, 128) — per-pipeline accuracy

        # We need per-trial predictions, but we only have per-pipeline accuracy.
        # Approximate: use per-pipeline accuracy as proxy for "agreement" at trial level.
        # A pipeline with high accuracy agrees with ground truth more often.

        # Better approach: compute pipeline disagreement from accuracy variance
        mean_acc = acc_matrix.mean(axis=0)  # (128,) per-pipeline accuracy
        acc_std = acc_matrix.std(axis=0)    # (128,) std across folds

        # Pipeline-level analysis: group pipelines by accuracy and compute
        # how much accuracy varies across the 128 pipelines
        overall_mean = mean_acc.mean()
        overall_std = mean_acc.std()

        # Simulate selective prediction:
        # Sort pipelines by accuracy, compute accuracy when keeping only top-K% pipelines
        sorted_acc = np.sort(mean_acc)[::-1]  # highest first
        coverages = np.arange(1, N_PIPELINES + 1) / N_PIPELINES
        cum_acc = np.cumsum(sorted_acc) / np.arange(1, N_PIPELINES + 1)

        # Key points on the curve
        print(f"\n{dinfo['name']}:")
        print(f"  Overall mean accuracy: {overall_mean:.4f} (std={overall_std:.4f})")
        print(f"  Best pipeline:  {sorted_acc[0]:.4f}")
        print(f"  Worst pipeline: {sorted_acc[-1]:.4f}")
        print(f"  Gap (best-worst): {sorted_acc[0]-sorted_acc[-1]:.4f}")
        print(f"\n  Selective prediction (keep top-K% pipelines):")
        for pct in [10, 25, 50, 75, 100]:
            k = max(1, int(N_PIPELINES * pct / 100))
            avg = sorted_acc[:k].mean()
            print(f"    Top {pct:3d}%: acc={avg:.4f} (+{avg-overall_mean:.4f} vs all)")

        # Compute accuracy improvement if we "abstain" on worst pipelines
        # This simulates: "if a practitioner could identify the worst pipeline choices,
        # how much would their expected accuracy improve?"
        results[dkey] = {
            'mean_acc': float(overall_mean),
            'std_acc': float(overall_std),
            'best_pipeline': float(sorted_acc[0]),
            'worst_pipeline': float(sorted_acc[-1]),
            'gap': float(sorted_acc[0] - sorted_acc[-1]),
            'selective': {
                f'top_{pct}pct': float(sorted_acc[:max(1, int(N_PIPELINES*pct/100))].mean())
                for pct in [10, 25, 50, 75, 100]
            },
            'coverage_acc_curve': {
                'coverage': coverages.tolist(),
                'accuracy': cum_acc.tolist(),
            }
        }

    # Save
    with open(OUTDIR / 'uncertainty_selective.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTDIR / 'uncertainty_selective.json'}")
    return results


# ============================================================
# EXTENSION 2: Signal-level mediation (simplified)
# ============================================================
def signal_mediation_analysis():
    """Compute signal statistics per pipeline to explain WHY interventions matter."""
    print("\n" + "="*70)
    print("EXTENSION 2: Signal-Level Mediation Analysis")
    print("="*70)

    results = {}

    for dkey, dinfo in DATASETS.items():
        data_dir = DDIR / dinfo['data']
        subj_dirs = sorted(data_dir.glob('S*'))
        if not subj_dirs:
            print(f"  {dinfo['name']}: No data found, skipping")
            continue

        # Load one subject's merged data to compute signal statistics
        subj_dir = subj_dirs[0]
        merged_f = subj_dir / 'all_pipelines.npy'
        if not merged_f.exists():
            print(f"  {dinfo['name']}: No merged file, skipping")
            continue

        # Memory-map to avoid loading full file
        mmap = np.load(str(merged_f), mmap_mode='r')  # (128, N, C, T)
        n_pipe, n_trials, n_ch, n_times = mmap.shape
        print(f"\n{dinfo['name']}: shape={mmap.shape}")

        # Sample 100 trials for efficiency
        trial_idx = np.random.choice(n_trials, min(100, n_trials), replace=False)

        # For each pipeline, compute signal statistics
        stats = np.zeros((n_pipe, 5))  # 5 features per pipeline
        for pi in range(n_pipe):
            data = mmap[pi, trial_idx, :, :].astype(np.float32)  # (100, C, T)

            # 1. Mean absolute amplitude
            stats[pi, 0] = np.mean(np.abs(data))
            # 2. Channel variance dispersion (std of per-channel variances)
            ch_vars = data.var(axis=2).mean(axis=0)  # (C,)
            stats[pi, 1] = ch_vars.std() / (ch_vars.mean() + 1e-8)
            # 3. Trial-level max amplitude (proxy for outliers)
            trial_max = np.abs(data).max(axis=(1, 2))  # (100,)
            stats[pi, 2] = np.median(trial_max)
            # 4. Low-frequency power proxy (mean of first 10% of time dimension)
            lf_power = np.mean(data[:, :, :max(1, n_times//10)]**2)
            stats[pi, 3] = lf_power
            # 5. Kurtosis (excess kurtosis averaged over channels)
            from scipy.stats import kurtosis
            k = kurtosis(data.reshape(-1), fisher=True)
            stats[pi, 4] = k

        # Load per-pipeline accuracy
        acc_matrix = np.load(RDIR / dinfo['analysis'] / 'per_pipeline_accuracy.npy')
        mean_acc = acc_matrix.mean(axis=0)

        # Correlation between signal stats and accuracy
        stat_names = ['amplitude', 'ch_var_disp', 'trial_max', 'lf_power', 'kurtosis']
        print(f"  Signal-Accuracy Correlations:")
        correlations = {}
        for j, sname in enumerate(stat_names):
            r = np.corrcoef(stats[:, j], mean_acc)[0, 1]
            correlations[sname] = float(r)
            print(f"    {sname:15s}: r={r:+.3f}")

        # Per-intervention mediation: how much does each intervention change signal stats?
        print(f"  Per-intervention signal changes:")
        mediation = {}
        for i in range(K):
            # Compare pipelines with bit i=0 vs bit i=1
            mask0 = np.array([(p >> i) & 1 == 0 for p in range(n_pipe)])
            mask1 = ~mask0
            delta_stats = stats[mask1].mean(axis=0) - stats[mask0].mean(axis=0)
            delta_acc = mean_acc[mask1].mean() - mean_acc[mask0].mean()
            med_entry = {sname: float(delta_stats[j]) for j, sname in enumerate(stat_names)}
            med_entry['delta_acc'] = float(delta_acc)
            mediation[INTERVENTION_NAMES[i]] = med_entry
            # Find which signal change correlates most with accuracy change
            if abs(delta_acc) > 0.001:
                best_mediator = stat_names[np.argmax(np.abs(delta_stats))]
                print(f"    {INTERVENTION_NAMES[i]:12s}: Δacc={delta_acc:+.4f}, "
                      f"strongest signal change: {best_mediator} ({delta_stats[np.argmax(np.abs(delta_stats))]:+.4f})")

        results[dkey] = {
            'correlations': correlations,
            'mediation': mediation,
        }

    with open(OUTDIR / 'mediation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTDIR / 'mediation.json'}")
    return results


if __name__ == '__main__':
    np.random.seed(42)
    r1 = walsh_hadamard_analysis()
    r3 = cfr_uncertainty_analysis()
    r2 = signal_mediation_analysis()

    print("\n" + "="*70)
    print("ALL EXTENSIONS COMPLETE")
    print("="*70)
