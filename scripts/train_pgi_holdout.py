#!/usr/bin/env python
"""
Exp 7: Held-out pipeline composition test.

Train PGI on 12 pipelines (remove 4 held-out corners of the hypercube),
test on all 16 including the 4 unseen compositions.
This tests whether edge invariance generalizes to unseen pipeline compositions.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.pgi import PGITrainer, ERMTrainer
from src.dataset import MultiPipelineDataset, create_subject_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(data_dir.glob("S*"))
    all_subjects = [int(d.name[1:]) for d in subject_dirs]

    # Hold out 4 pipeline corners (indices with all bits set in different combos)
    # Hold out: 3 (0011), 5 (0101), 10 (1010), 12 (1100) — diverse corners
    holdout_pipelines = [3, 5, 10, 12]
    train_pipelines = [i for i in range(16) if i not in holdout_pipelines]
    logger.info(f"Train pipelines: {train_pipelines}")
    logger.info(f"Holdout pipelines: {holdout_pipelines}")

    splits = create_subject_splits(len(all_subjects), n_folds=3, seed=args.seed)
    results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_subjs = [all_subjects[i] for i in train_idx]
        val_subjs = [all_subjects[i] for i in val_idx]
        test_subjs = [all_subjects[i] for i in test_idx]

        logger.info(f"\n=== Fold {fold_idx} ===")
        logger.info(f"Train: {train_subjs}, Val: {val_subjs}, Test: {test_subjs}")

        train_data, train_labels = load_dataset(data_dir, train_subjs)
        val_data, val_labels = load_dataset(data_dir, val_subjs)
        test_data, test_labels = load_dataset(data_dir, test_subjs)

        # Train dataset: only seen pipelines
        train_ds = MultiPipelineDataset(train_data, train_labels,
                                         pipeline_indices=train_pipelines)
        # Val/Test: ALL 16 pipelines (to measure generalization)
        val_ds = MultiPipelineDataset(val_data, val_labels)
        test_ds_all = MultiPipelineDataset(test_data, test_labels)
        # Also test on holdout-only
        test_ds_holdout = MultiPipelineDataset(test_data, test_labels,
                                                pipeline_indices=holdout_pipelines)
        test_ds_seen = MultiPipelineDataset(test_data, test_labels,
                                             pipeline_indices=train_pipelines)

        n_channels = train_ds.views.shape[2]
        n_times = train_ds.views.shape[3]
        n_classes = len(np.unique(train_labels))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=2,
                                   pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                 num_workers=2, pin_memory=True)

        for method_name in ["pgi", "erm_mixed"]:
            model = get_model(args.model, n_channels=n_channels,
                              n_times=n_times, n_classes=n_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                           weight_decay=1e-4)

            if method_name == "pgi":
                # PGI only sees edges between train_pipelines
                trainer = PGITrainer(model, optimizer, n_classes,
                                      pgi_lambda=args.pgi_lambda, device=device,
                                      pipeline_indices=train_pipelines)
            else:
                trainer = ERMTrainer(model, optimizer, n_classes,
                                      mixed=True, device=device)

            best_val_acc = 0
            for epoch in range(args.epochs):
                for bv, bl in train_loader:
                    bv, bl = bv.to(device), bl.to(device)
                    trainer.train_step(bv, bl, epoch=epoch)

                if epoch % 20 == 0 or epoch == args.epochs - 1:
                    vm = []
                    for bv, bl in val_loader:
                        bv, bl = bv.to(device), bl.to(device)
                        vm.append(trainer.eval_step(bv, bl))
                    va = np.mean([m["mean_accuracy"] for m in vm])
                    if va > best_val_acc:
                        best_val_acc = va
                        torch.save(model.state_dict(),
                                   out_dir / f"best_{method_name}_f{fold_idx}.pt")

            # Evaluate on all, seen, holdout
            model.load_state_dict(torch.load(
                out_dir / f"best_{method_name}_f{fold_idx}.pt", weights_only=True))
            model.eval()

            def eval_ds(ds):
                loader = DataLoader(ds, batch_size=args.batch_size,
                                     num_workers=2, pin_memory=True)
                metrics = []
                for bv, bl in loader:
                    bv, bl = bv.to(device), bl.to(device)
                    metrics.append(trainer.eval_step(bv, bl))
                return {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}

            r_all = eval_ds(test_ds_all)
            r_seen = eval_ds(test_ds_seen)
            r_holdout = eval_ds(test_ds_holdout)

            logger.info(f"\n{method_name} Fold {fold_idx}:")
            logger.info(f"  All 16:   acc={r_all['mean_accuracy']:.3f} CFR={r_all['cfr']:.3f}")
            logger.info(f"  Seen 12:  acc={r_seen['mean_accuracy']:.3f} CFR={r_seen['cfr']:.3f}")
            logger.info(f"  Holdout 4: acc={r_holdout['mean_accuracy']:.3f} CFR={r_holdout['cfr']:.3f}")

            gap = r_seen["mean_accuracy"] - r_holdout["mean_accuracy"]
            logger.info(f"  Seen-Holdout gap: {gap:.3f}")

            results.append({
                "fold": fold_idx, "method": method_name,
                "all": r_all, "seen": r_seen, "holdout": r_holdout,
                "gap": gap,
            })

    with open(out_dir / "holdout_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to {out_dir / 'holdout_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/holdout")
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pgi-lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
