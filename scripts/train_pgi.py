#!/usr/bin/env python
"""
Main training script for PGI experiments.

Supports: PGI, ERM-single, ERM-mixed, PairedConsistency.
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
from src.pgi import PGITrainer, ERMTrainer, PairedConsistencyTrainer
from src.dataset import MultiPipelineDataset, create_subject_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine subjects
    subject_dirs = sorted(data_dir.glob("S*"))
    all_subjects = [int(d.name[1:]) for d in subject_dirs]
    logger.info(f"Found {len(all_subjects)} subjects: {all_subjects}")

    splits = create_subject_splits(len(all_subjects), n_folds=args.n_folds,
                                    seed=args.seed)

    results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        if args.fold is not None and fold_idx != args.fold:
            continue

        train_subjs = [all_subjects[i] for i in train_idx]
        val_subjs = [all_subjects[i] for i in val_idx]
        test_subjs = [all_subjects[i] for i in test_idx]

        logger.info(f"\n=== Fold {fold_idx} ===")
        logger.info(f"Train: {train_subjs}, Val: {val_subjs}, Test: {test_subjs}")

        # Lazy-loading datasets (no RAM explosion with 128 pipelines)
        # PGI loads all views; other methods subsample
        sv = None if args.method == "pgi" else args.n_views_per_batch
        train_ds = MultiPipelineDataset(
            data_dir, train_subjs, n_pipelines=args.n_pipelines,
            n_views_sample=sv
        )
        val_ds = MultiPipelineDataset(
            data_dir, val_subjs, n_pipelines=args.n_pipelines
        )
        test_ds = MultiPipelineDataset(
            data_dir, test_subjs, n_pipelines=args.n_pipelines
        )

        # Get data dimensions from first sample
        sample_views, _ = train_ds[0]
        n_channels = sample_views.shape[1]
        n_times = sample_views.shape[2]
        n_classes = len(np.unique(train_ds.labels))
        logger.info(f"Data: {n_channels}ch, {n_times}t, {n_classes}cls, "
                     f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=2,
                                   pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        # Model
        model = get_model(args.model, n_channels=n_channels,
                          n_times=n_times, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)

        # Trainer
        if args.method == "pgi":
            trainer = PGITrainer(
                model, optimizer, n_classes,
                pgi_lambda=args.pgi_lambda,
                edge_weight_mode=args.edge_weight_mode,
                warmup_epochs=args.pgi_warmup_epochs,
                device=device,
                normalize_pgi=args.normalize_pgi,
                adaptive_lambda=args.adaptive_lambda,
                cfr_target=args.cfr_target,
            )
        elif args.method == "erm_single":
            trainer = ERMTrainer(model, optimizer, n_classes,
                                  mixed=False, device=device)
        elif args.method == "erm_mixed":
            trainer = ERMTrainer(model, optimizer, n_classes,
                                  mixed=True, device=device)
        elif args.method == "consistency":
            trainer = PairedConsistencyTrainer(
                model, optimizer, n_classes,
                consistency_lambda=args.pgi_lambda, device=device,
            )
        elif args.method == "groupdro":
            from src.pgi import GroupDROTrainer
            trainer = GroupDROTrainer(model, optimizer, n_classes,
                                      n_views_sample=args.n_views_per_batch,
                                      device=device)
        elif args.method == "irm":
            from src.pgi import IRMTrainer
            trainer = IRMTrainer(model, optimizer, n_classes,
                                  irm_lambda=args.pgi_lambda,
                                  n_views_sample=args.n_views_per_batch,
                                  device=device)
        elif args.method == "coral":
            from src.pgi import CORALTrainer
            trainer = CORALTrainer(model, optimizer, n_classes,
                                    coral_lambda=args.pgi_lambda,
                                    n_views_sample=args.n_views_per_batch,
                                    device=device)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        # Training loop
        best_val_acc = 0
        best_epoch = 0

        for epoch in range(args.epochs):
            t0 = time.time()
            train_metrics = []

            for batch_views, batch_labels in train_loader:
                batch_views = batch_views.to(device)
                batch_labels = batch_labels.to(device)
                if args.method == "pgi":
                    m = trainer.train_step(batch_views, batch_labels, epoch=epoch,
                                           n_edge_sample=args.n_edge_sample,
                                           n_sup_views=args.n_sup_views)
                else:
                    m = trainer.train_step(batch_views, batch_labels, epoch=epoch)
                train_metrics.append(m)

            # Average train metrics
            avg_train = {}
            for key in train_metrics[0]:
                avg_train[key] = np.mean([m[key] for m in train_metrics])

            # Validation
            val_metrics = []
            for batch_views, batch_labels in val_loader:
                batch_views = batch_views.to(device)
                batch_labels = batch_labels.to(device)
                m = trainer.eval_step(batch_views, batch_labels)
                val_metrics.append(m)

            avg_val = {}
            for key in val_metrics[0]:
                avg_val[key] = np.mean([m[key] for m in val_metrics])

            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train loss={avg_train.get('loss', 0):.4f} "
                    f"acc={avg_train.get('accuracy', 0):.3f} | "
                    f"Val acc={avg_val.get('mean_accuracy', 0):.3f} "
                    f"worst={avg_val.get('worst_accuracy', 0):.3f} "
                    f"CFR={avg_val.get('cfr', 0):.3f} | "
                    f"{elapsed:.1f}s"
                )

            # Save best model
            if avg_val.get("mean_accuracy", 0) > best_val_acc:
                best_val_acc = avg_val["mean_accuracy"]
                best_epoch = epoch
                torch.save(model.state_dict(),
                           out_dir / f"best_fold{fold_idx}.pt")

        # Test evaluation with best model
        model.load_state_dict(torch.load(out_dir / f"best_fold{fold_idx}.pt",
                                          weights_only=True))
        test_metrics = []
        for batch_views, batch_labels in test_loader:
            batch_views = batch_views.to(device)
            batch_labels = batch_labels.to(device)
            m = trainer.eval_step(batch_views, batch_labels)
            test_metrics.append(m)

        avg_test = {}
        for key in test_metrics[0]:
            avg_test[key] = np.mean([m[key] for m in test_metrics])

        logger.info(
            f"\nFold {fold_idx} Test Results (best epoch {best_epoch}):\n"
            f"  Mean Acc: {avg_test.get('mean_accuracy', 0):.4f}\n"
            f"  Worst Acc: {avg_test.get('worst_accuracy', 0):.4f}\n"
            f"  CFR: {avg_test.get('cfr', 0):.4f}\n"
            f"  Max CFR: {avg_test.get('max_cfr', 0):.4f}"
        )

        fold_result = {
            "fold": fold_idx,
            "method": args.method,
            "model": args.model,
            "best_epoch": best_epoch,
            "test": avg_test,
            "seed": args.seed,
        }
        results.append(fold_result)

    # Save results
    results_file = out_dir / f"results_{args.method}_{args.model}_seed{args.seed}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Preprocessed data dir")
    parser.add_argument("--output-dir", default="results", help="Output dir")
    parser.add_argument("--method", default="pgi",
                        choices=["pgi", "erm_single", "erm_mixed", "consistency",
                                 "groupdro", "irm", "coral"])
    parser.add_argument("--model", default="eegnet",
                        choices=["eegnet", "shallow"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pgi-lambda", type=float, default=1.0)
    parser.add_argument("--pgi-warmup-epochs", type=int, default=5)
    parser.add_argument("--edge-weight-mode", default="perturbation",
                        choices=["uniform", "perturbation"])
    parser.add_argument("--n-views-per-batch", type=int, default=8)
    parser.add_argument("--n-edge-sample", type=int, default=32,
                        help="Edges to sample per PGI step (0=all)")
    parser.add_argument("--n-sup-views", type=int, default=8,
                        help="Views for supervised loss per step (0=all)")
    parser.add_argument("--n-pipelines", type=int, default=128,
                        help="Number of pipelines in the dataset")
    parser.add_argument("--normalize-pgi", action="store_true", default=False,
                        help="Normalize PGI loss by logit variance (scale-invariant)")
    parser.add_argument("--adaptive-lambda", action="store_true", default=False,
                        help="Adaptive lambda based on running CFR")
    parser.add_argument("--cfr-target", type=float, default=0.15,
                        help="Target CFR for adaptive lambda")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Run single fold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
