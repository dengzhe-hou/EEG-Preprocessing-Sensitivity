#!/usr/bin/env python
"""
Per-intervention sensitivity analysis (Block B2/B4).

Loads a trained model, runs inference on all 128 pipelines,
computes per-intervention accuracy effect and CFR contribution.
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.dataset import MultiPipelineDataset, create_subject_splits
from src.intervention_graph import (
    INTERVENTION_NAMES, K, N_PIPELINES, get_intervention_edges
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(data_dir.glob("S*"))
    all_subjects = [int(d.name[1:]) for d in subject_dirs]
    splits = create_subject_splits(len(all_subjects), n_folds=3, seed=42)

    all_per_pipeline_acc = []  # (n_folds, 128) per-pipeline accuracy

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        test_subjs = [all_subjects[i] for i in test_idx]
        logger.info(f"Fold {fold_idx}: test subjects {test_subjs}")

        # Load test data
        test_ds = MultiPipelineDataset(data_dir, test_subjs, n_pipelines=128)
        sample, _ = test_ds[0]
        n_ch, n_t = sample.shape[1], sample.shape[2]
        n_cls = len(np.unique(test_ds.labels))

        # Load trained model
        model_path = Path(args.model_dir) / f"best_fold{fold_idx}.pt"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        model = get_model("eegnet", n_channels=n_ch, n_times=n_t,
                          n_classes=n_cls).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True,
                                          map_location=device))
        model.eval()

        # Run inference on all 128 pipelines
        per_pipeline_correct = np.zeros(128)
        per_pipeline_total = np.zeros(128)
        per_pipeline_preds = {pi: [] for pi in range(128)}

        with torch.no_grad():
            for idx in range(len(test_ds)):
                views, label = test_ds[idx]  # (128, C, T)
                views = views.to(device)
                label = label.item()

                # Forward each pipeline separately to save memory
                for pi in range(128):
                    logits = model(views[pi:pi+1])
                    pred = logits.argmax(-1).item()
                    per_pipeline_preds[pi].append(pred)
                    if pred == label:
                        per_pipeline_correct[pi] += 1
                    per_pipeline_total[pi] += 1

                if idx % 100 == 0:
                    logger.info(f"  Fold {fold_idx}: {idx}/{len(test_ds)}")

        per_pipeline_acc = per_pipeline_correct / np.maximum(per_pipeline_total, 1)
        all_per_pipeline_acc.append(per_pipeline_acc)

    # Average across folds
    acc_matrix = np.stack(all_per_pipeline_acc)  # (n_folds, 128)
    mean_acc = acc_matrix.mean(axis=0)  # (128,)

    # Per-intervention analysis
    print("\n" + "="*70)
    print("Per-Intervention Sensitivity Analysis")
    print("="*70)
    print(f"\n{'Intervention':15s} | {'Δ Accuracy':>10s} | {'|Δ|':>6s} | {'Direction':>9s}")
    print("-"*55)

    results = {}
    for i in range(K):
        pairs = get_intervention_edges(i)
        deltas = []
        for src, dst in pairs:
            # src has bit i=0, dst has bit i=1
            delta = mean_acc[dst] - mean_acc[src]
            deltas.append(delta)

        mean_delta = np.mean(deltas)
        abs_delta = np.mean(np.abs(deltas))
        direction = "+" if mean_delta > 0 else "-"

        print(f"{INTERVENTION_NAMES[i]:15s} | {mean_delta:+10.4f} | {abs_delta:6.4f} | {direction:>9s}")

        results[INTERVENTION_NAMES[i]] = {
            "mean_delta": float(mean_delta),
            "abs_delta": float(abs_delta),
            "std_delta": float(np.std(deltas)),
            "n_positive": int(sum(1 for d in deltas if d > 0)),
            "n_negative": int(sum(1 for d in deltas if d < 0)),
        }

    # Rank by absolute effect
    print(f"\nRanking by |Δ| (most → least impactful):")
    ranked = sorted(results.items(), key=lambda x: x[1]["abs_delta"], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank}. {name:15s}: |Δ|={r['abs_delta']:.4f}, "
              f"sign={'+' if r['mean_delta']>0 else '-'}, "
              f"({r['n_positive']}+/{r['n_negative']}-)")

    # Save
    with open(out_dir / "intervention_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(out_dir / "per_pipeline_accuracy.npy", acc_matrix)
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Dir with best_fold{i}.pt files")
    parser.add_argument("--output-dir", default="results_v4/analysis")
    main(parser.parse_args())
