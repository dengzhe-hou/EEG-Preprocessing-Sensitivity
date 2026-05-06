#!/usr/bin/env python
"""
Exp 2: Embedding-space pipeline fingerprinting.

Train a task model on one pipeline, freeze it, extract embeddings for all
16 pipelines, then probe for pipeline identity from frozen embeddings.
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


def load_dataset(data_dir, subjects):
    all_data = {i: [] for i in range(16)}
    all_labels = []
    for subj in subjects:
        subj_dir = data_dir / f"S{subj:02d}"
        if not subj_dir.exists():
            continue
        labels = np.load(subj_dir / "labels.npy")
        all_labels.append(labels)
        for pi in range(16):
            f = subj_dir / f"pipeline_{pi:02d}.npz"
            if f.exists():
                all_data[pi].append(np.load(f)["data"])
    for pi in all_data:
        if all_data[pi]:
            all_data[pi] = np.concatenate(all_data[pi], axis=0)
    return all_data, np.concatenate(all_labels, axis=0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(data_dir.glob("S*"))
    all_subjects = [int(d.name[1:]) for d in subject_dirs]

    splits = create_subject_splits(len(all_subjects), n_folds=3, seed=args.seed)
    results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_subjs = [all_subjects[i] for i in train_idx]
        test_subjs = [all_subjects[i] for i in test_idx]

        logger.info(f"\n=== Fold {fold_idx} ===")

        train_data, train_labels = load_dataset(data_dir, train_subjs)
        test_data, test_labels = load_dataset(data_dir, test_subjs)

        # Step 1: Train task model on pipeline 0 only
        min_t = min(train_data[pi].shape[-1] for pi in range(16) if len(train_data[pi]) > 0)
        X_train = train_data[0][..., :min_t].astype(np.float32)
        X_train = (X_train - X_train.mean(-1, keepdims=True)) / (X_train.std(-1, keepdims=True) + 1e-8)

        n_channels, n_times = X_train.shape[1], X_train.shape[2]
        n_classes = len(np.unique(train_labels))

        model = get_model("eegnet", n_channels=n_channels, n_times=n_times,
                          n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        X_t = torch.tensor(X_train).to(device)
        y_t = torch.tensor(train_labels).to(device)

        logger.info(f"Training task model on pipeline 0... ({len(X_train)} samples)")
        model.train()
        for epoch in range(50):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), 128):
                idx = perm[i:i+128]
                logits = model(X_t[idx])
                loss = F.cross_entropy(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Step 2: Freeze model, extract embeddings for all 16 pipelines
        model.eval()
        all_embeddings = []
        all_pipe_labels = []

        for pi in range(16):
            X_pi = test_data[pi][..., :min_t].astype(np.float32)
            X_pi = (X_pi - X_pi.mean(-1, keepdims=True)) / (X_pi.std(-1, keepdims=True) + 1e-8)
            X_pi_t = torch.tensor(X_pi).to(device)

            with torch.no_grad():
                feats = []
                for i in range(0, len(X_pi_t), 128):
                    feats.append(model.get_features(X_pi_t[i:i+128]))
                feats = torch.cat(feats).cpu().numpy()

            all_embeddings.append(feats)
            all_pipe_labels.append(np.full(len(feats), pi, dtype=np.int64))

        embeddings = np.concatenate(all_embeddings, axis=0)
        pipe_labels = np.concatenate(all_pipe_labels, axis=0)

        # Step 3: Train linear probe for pipeline identity
        n = len(embeddings)
        perm = np.random.permutation(n)
        split = int(0.7 * n)
        train_emb, test_emb = embeddings[perm[:split]], embeddings[perm[split:]]
        train_pl, test_pl = pipe_labels[perm[:split]], pipe_labels[perm[split:]]

        probe = nn.Linear(embeddings.shape[1], 16).to(device)
        probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        train_emb_t = torch.tensor(train_emb).to(device)
        train_pl_t = torch.tensor(train_pl).to(device)
        test_emb_t = torch.tensor(test_emb).to(device)

        for ep in range(50):
            probe.train()
            perm_p = torch.randperm(len(train_emb_t))
            for i in range(0, len(train_emb_t), 256):
                idx = perm_p[i:i+256]
                logits = probe(train_emb_t[idx])
                loss = F.cross_entropy(logits, train_pl_t[idx])
                probe_opt.zero_grad()
                loss.backward()
                probe_opt.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_emb_t)
            test_preds = test_logits.argmax(-1).cpu().numpy()
            acc = balanced_accuracy_score(test_pl, test_preds)

        # Step 4: Within-class probe (control)
        # Use task labels to check if pipeline is detectable within each class
        task_labels_test = np.tile(test_labels, 16)[perm[split:]]
        within_class_accs = []
        for cls in range(n_classes):
            mask = task_labels_test == cls
            if mask.sum() > 100:
                cls_acc = balanced_accuracy_score(test_pl[mask], test_preds[mask])
                within_class_accs.append(cls_acc)

        within_acc = np.mean(within_class_accs) if within_class_accs else 0

        logger.info(f"Fold {fold_idx}: Embedding probe acc = {acc:.4f} "
                    f"(within-class = {within_acc:.4f}, chance = 0.0625)")

        results.append({
            "fold": fold_idx,
            "embedding_probe_acc": float(acc),
            "within_class_probe_acc": float(within_acc),
            "chance": 1/16,
        })

    avg_acc = np.mean([r["embedding_probe_acc"] for r in results])
    avg_wc = np.mean([r["within_class_probe_acc"] for r in results])
    logger.info(f"\nExp 2 Summary: Embedding probe = {avg_acc:.4f}, "
               f"Within-class = {avg_wc:.4f}, Chance = 0.0625")

    with open(out_dir / "embedding_fingerprint_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/embedding_fp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
