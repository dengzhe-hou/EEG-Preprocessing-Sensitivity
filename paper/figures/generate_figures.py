#!/usr/bin/env python
"""Generate all data figures for the NeurIPS paper (6 datasets)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/hou/Research/Unified-EEG-Preprocessing')

# Style
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

OUTDIR = '/home/hou/Research/Unified-EEG-Preprocessing/paper/figures'
RDIR = '/home/hou/Research/Unified-EEG-Preprocessing/results_v4'

INTERVENTIONS = ['reference', 'hpf', 'lpf', 'baseline', 'asr', 'autoreject', 'badchannel']
NICE_NAMES = ['Reference', 'High-pass', 'Low-pass', 'Baseline', 'ASR', 'Epoch reject', 'Bad channel']

# All 6 datasets — ordered by CFR (high to low) matching Tab:cfr
DS_KEYS = ['bci', 'seed_iv', 'physionet', 'sleep', 'lee2019_erp', 'p300']
DS_SHORT = ['BCI (MI)', 'SEED-IV', 'PhysionetMI', 'Sleep', 'Lee2019', 'P300']
DS_FULL = [
    'BCI-IV-2a\n(MI, 4-cls)',
    'SEED-IV\n(Emo, 4-cls)',
    'PhysionetMI\n(MI, 2-cls)',
    'Sleep-EDF\n(Sleep, 5-cls)',
    'Lee2019\n(ERP, 2-cls)',
    'P300\n(ERP, 2-cls)',
]

# CFR and accuracy from Tab:cfr (verified against experiment data)
CFR_VALUES = [42.4, 35.8, 21.7, 9.6, 4.1, 2.6]
ACC_VALUES = [37.6, 31.9, 57.7, 85.6, 84.1, 83.4]

# Load analysis data for all 6 datasets
analysis = {}
for key in DS_KEYS:
    path = f'{RDIR}/analysis_{key}/intervention_analysis.json'
    analysis[key] = json.load(open(path))

# ============================================================
# FIGURE 1: CFR summary bar chart (6 datasets)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 2.8))

x = np.arange(len(DS_FULL))
width = 0.35

bars1 = ax.bar(x - width/2, ACC_VALUES, width, label='Mean Accuracy (%)',
               color='#4C72B0', alpha=0.8)
bars2 = ax.bar(x + width/2, CFR_VALUES, width, label='CFR (%)',
               color='#DD8452', alpha=0.8)

ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(DS_FULL, fontsize=7)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, 100)
ax.set_title('Preprocessing Sensitivity vs Task Accuracy', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig1_cfr_summary.pdf')
plt.savefig(f'{OUTDIR}/fig1_cfr_summary.png')
plt.savefig(f'{OUTDIR}/fig1_cfr_summary.svg')
plt.close()
print('Fig 1: CFR summary (6 datasets) saved')

# ============================================================
# FIGURE 2: Intervention heatmap 7x6 (absolute effects)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))

data_matrix = np.zeros((7, 6))
for i, intv in enumerate(INTERVENTIONS):
    for j, key in enumerate(DS_KEYS):
        data_matrix[i, j] = analysis[key][intv]['abs_delta'] * 100

# Sort by max effect across datasets
max_effect = data_matrix.max(axis=1)
sort_idx = np.argsort(-max_effect)
data_sorted = data_matrix[sort_idx]
names_sorted = [NICE_NAMES[i] for i in sort_idx]

sns.heatmap(data_sorted, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=DS_SHORT, yticklabels=names_sorted,
            ax=ax, cbar_kws={'label': '|Δ Accuracy| (%)', 'shrink': 0.8},
            vmin=0, vmax=10,
            linewidths=0.5, linecolor='white')
ax.set_title('Per-Intervention Sensitivity Across Tasks', fontsize=10, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig2_intervention_heatmap.pdf')
plt.savefig(f'{OUTDIR}/fig2_intervention_heatmap.png')
plt.savefig(f'{OUTDIR}/fig2_intervention_heatmap.svg')
plt.close()
print('Fig 2: intervention heatmap (7x6) saved')

# ============================================================
# FIGURE 3: Signed effects heatmap 7x6
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.2))

signed_matrix = np.zeros((7, 6))
for i, intv in enumerate(INTERVENTIONS):
    for j, key in enumerate(DS_KEYS):
        signed_matrix[i, j] = analysis[key][intv]['mean_delta'] * 100

signed_sorted = signed_matrix[sort_idx]

sns.heatmap(signed_sorted, annot=True, fmt='+.1f', cmap='RdBu_r',
            center=0, vmin=-10, vmax=10,
            xticklabels=DS_SHORT, yticklabels=names_sorted,
            ax=ax, cbar_kws={'label': 'Δ Accuracy (%)', 'shrink': 0.8},
            linewidths=0.5, linecolor='white')
ax.set_title('Signed Effect of Each Intervention (positive = helps)', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig3_signed_effects.pdf')
plt.savefig(f'{OUTDIR}/fig3_signed_effects.png')
plt.savefig(f'{OUTDIR}/fig3_signed_effects.svg')
plt.close()
print('Fig 3: signed effects (7x6) saved')

print('\nAll figures generated!')
