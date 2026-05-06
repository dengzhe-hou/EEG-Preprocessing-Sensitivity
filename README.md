# Same Brain, Different Prediction

**How Preprocessing Choices Undermine EEG Decoding Reliability**

## Key Finding

Up to **42% of EEG trial-level predictions flip** when only the preprocessing pipeline changes — the model, data, and labels remain identical. This variability is invisible to standard uncertainty methods (softmax entropy, MC Dropout).

## Contributions

1. **Factorial analysis**: A Walsh–Hadamard decomposition of the 2⁷ pipeline space reveals that preprocessing sensitivity is predominantly additive across six datasets and four EEG paradigms.
2. **Preprocessing Uncertainty (PU)**: A per-trial diagnostic that quantifies pipeline disagreement, complementary to model-based confidence.
3. **NA-PGI**: A graph-structured regularizer with logit-variance normalization and adaptive λ scaling that reduces CFR by up to 35% with a single hyperparameter (λ=1).

## Project Structure

```
src/
├── intervention_graph.py   # 7 interventions, 128 pipelines, 448 Hasse edges
├── preprocessing.py        # MNE-based preprocessing pipeline generator
├── dataset.py              # Memory-mapped multi-pipeline dataset
├── models.py               # EEGNet, ShallowNet
└── pgi.py                  # All trainers: PGI, ERM, Consistency, GroupDRO, IRM, CORAL

scripts/
├── prepare_*_v4.py         # Data preparation for 6 datasets
├── train_pgi.py            # Main training script (7 methods)
├── analyze_interventions.py # Per-intervention sensitivity analysis
├── analyze_extensions.py   # Walsh-Hadamard, signal mediation, selective prediction
├── compute_pu.py           # PU framework
└── validate_additivity.py  # Greedy vs oracle validation
```

## Datasets

| Dataset | Task | Classes | Channels | Source |
|---------|------|---------|----------|--------|
| BCI-IV-2a | Motor Imagery | 4 | 22 | MOABB |
| PhysionetMI | Motor Imagery | 2 | 64 | MOABB |
| Sleep-EDF | Sleep Staging | 5 | 2 | PhysioNet |
| BNCI2014-009 | P300 (ERP) | 2 | 16 | MOABB |
| Lee2019-ERP | P300 (ERP) | 2 | 62 | MOABB |
| SEED-IV | Emotion | 4 | 62 | BCMI |

## Quick Start

```bash
# Install dependencies
pip install torch mne moabb numpy scipy

# Prepare data (example: BCI-IV-2a)
python scripts/prepare_bci_v4.py

# Train ERM baseline
python scripts/train_pgi.py \
    --data-dir data/processed_v4/bci_iv_2a \
    --method erm_single --model eegnet --seed 42

# Train NA-PGI
python scripts/train_pgi.py \
    --data-dir data/processed_v4/bci_iv_2a \
    --method pgi --model eegnet --seed 42 \
    --normalize-pgi --adaptive-lambda

# Run sensitivity analysis
python scripts/analyze_interventions.py
python scripts/analyze_extensions.py
```

## License

MIT
