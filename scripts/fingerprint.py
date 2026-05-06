#!/usr/bin/env python
"""
Exp 1-2: Pipeline fingerprinting.

Exp 1: Train a classifier to predict pipeline identity from processed EEG.
Exp 2: Train a task model, freeze it, then probe for pipeline identity
       from its embeddings.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.dataset import MultiPipelineDataset, create_subject_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class PipelineProbe(nn.Module):
    """Linear or MLP probe for pipeline identity."""

    def __init__(self, input_dim, n_pipelines=16, n_layers=1, hidden=128):
        super().__init__()
        if n_layers == 1:
            self.net = nn.Linear(input_dim, n_pipelines)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, n_pipelines),
            )

    def forward(self, x):
        return self.net(x)


def signal_space_fingerprint(data_dir, subjects, n_channels, n_times, device):
    """Exp 1: Classify pipeline identity from processed EEG signals."""
    logger.info("=== Exp 1: Signal-Space Fingerprinting ===")

    # Load all data
    all_X, all_pipeline_labels = [], []
    for subj in subjects:
        subj_dir = data_dir / f"S{subj:02d}"
        for pi in range(16):
            f = subj_dir / f"pipeline_{pi:02d}.npz"
            if f.exists():
                d = np.load(f)["data"].astype(np.float32)
                # Z-score
                d = (d - d.mean(-1, keepdims=True)) / (d.std(-1, keepdims=True) + 1e-8)
                all_X.append(d)
                all_pipeline_labels.append(np.full(len(d), pi, dtype=np.int64))

    # Truncate all to minimum time dimension (pipelines may differ due to resampling)
    min_t = min(a.shape[-1] for a in all_X)
    all_X = [a[..., :min_t] for a in all_X]
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_pipeline_labels, axis=0)
    logger.info(f"Total samples: {len(X)}, shape: {X.shape} (truncated to T={min_t})")

    # Simple train/test split (80/20)
    n = len(X)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    # Use a simple EEGNet as the fingerprint classifier
    model = get_model("eegnet", n_channels=n_channels, n_times=X.shape[-1],
                      n_classes=16, F1=4, D=1, F2=8, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train = torch.tensor(X[train_idx]).to(device)
    y_train = torch.tensor(y[train_idx]).to(device)
    X_test = torch.tensor(X[test_idx]).to(device)
    y_test = torch.tensor(y[test_idx]).to(device)

    # Train
    batch_size = 128
    for epoch in range(30):
        model.train()
        perm_e = torch.randperm(len(X_train))
        losses = []
        for i in range(0, len(X_train), batch_size):
            idx = perm_e[i:i + batch_size]
            logits = model(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = []
                for i in range(0, len(X_test), batch_size):
                    test_logits.append(model(X_test[i:i + batch_size]))
                test_logits = torch.cat(test_logits)
                test_preds = test_logits.argmax(-1).cpu().numpy()
                test_acc = balanced_accuracy_score(y_test.cpu().numpy(), test_preds)
            logger.info(f"Epoch {epoch}: loss={np.mean(losses):.4f}, "
                       f"test_acc={test_acc:.4f} (chance=6.25%)")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = []
        for i in range(0, len(X_test), batch_size):
            test_logits.append(model(X_test[i:i + batch_size]))
        test_logits = torch.cat(test_logits)
        test_preds = test_logits.argmax(-1).cpu().numpy()
        final_acc = balanced_accuracy_score(y_test.cpu().numpy(), test_preds)

    logger.info(f"\nExp 1 Result: Pipeline fingerprint accuracy = {final_acc:.4f} "
               f"(chance = {1/16:.4f})")
    return {"signal_fingerprint_acc": final_acc, "chance": 1 / 16}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get subjects
    subject_dirs = sorted(data_dir.glob("S*"))
    subjects = [int(d.name[1:]) for d in subject_dirs]

    # Get data dimensions
    sample = np.load(subject_dirs[0] / "pipeline_00.npz")["data"]
    n_channels, n_times = sample.shape[1], sample.shape[2]

    results = {}

    # Exp 1: Signal-space fingerprinting
    r1 = signal_space_fingerprint(data_dir, subjects, n_channels, n_times, device)
    results["exp1"] = r1

    # Save results
    with open(out_dir / "fingerprint_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_dir / 'fingerprint_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/fingerprint")
    args = parser.parse_args()
    main(parser.parse_args())
