#!/usr/bin/env python
"""
Preprocessing Uncertainty (PU) Framework.

For each test trial, compute:
1. PU = 1 - max_c(fraction of 128 pipelines predicting class c)
2. PE = entropy of prediction distribution across pipelines
3. Softmax entropy from pipeline-0 (standard baseline)
4. MC Dropout uncertainty (30 passes with dropout enabled)

Then evaluate: calibration, abstention curves, correlation, AUROC for error detection.
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.dataset import MultiPipelineDataset, create_subject_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def compute_trial_predictions(model, dataset, device, n_mc_dropout=30):
    """Run inference on all 128 pipelines + MC Dropout for each test trial.

    Returns:
        preds: (N, 128) int — predicted class per pipeline per trial
        logits_p0: (N, C) float — logits from pipeline 0 (for softmax entropy)
        mc_preds: (N, n_mc) int — MC Dropout predictions (pipeline 0, dropout ON)
        labels: (N,) int — ground truth
    """
    model.eval()
    all_preds = []
    all_logits_p0 = []
    all_mc_preds = []
    all_labels = []

    for idx in range(len(dataset)):
        views, label = dataset[idx]  # views: (128, C, T)
        views = views.to(device)
        n_views = views.shape[0]

        # 1. Predictions from all 128 pipelines (model in eval mode)
        with torch.no_grad():
            trial_preds = []
            for pi in range(n_views):
                logits = model(views[pi:pi+1])
                trial_preds.append(logits.argmax(-1).item())
                if pi == 0:
                    all_logits_p0.append(logits.cpu().numpy()[0])
            all_preds.append(trial_preds)

        # 2. MC Dropout predictions (pipeline 0 only, dropout ON)
        model.train()  # enable dropout
        mc_trial = []
        with torch.no_grad():
            for _ in range(n_mc_dropout):
                logits = model(views[0:1])
                mc_trial.append(logits.argmax(-1).item())
        model.eval()
        all_mc_preds.append(mc_trial)

        all_labels.append(label.item())

        if idx % 200 == 0:
            logger.info(f"  {idx}/{len(dataset)}")

    return (np.array(all_preds), np.array(all_logits_p0),
            np.array(all_mc_preds), np.array(all_labels))


def compute_pu(preds, n_classes):
    """Compute Preprocessing Uncertainty for each trial.

    PU(x_i) = 1 - max_c(fraction of pipelines predicting c)

    Args:
        preds: (N, P) int array of predictions across P pipelines
        n_classes: number of classes

    Returns:
        pu: (N,) float array
    """
    N, P = preds.shape
    pu = np.zeros(N)
    for i in range(N):
        counts = np.bincount(preds[i], minlength=n_classes)
        pu[i] = 1.0 - counts.max() / P
    return pu


def compute_pe(preds, n_classes):
    """Compute Preprocessing Entropy for each trial.

    PE(x_i) = -sum_c q_c log(q_c), where q_c = fraction of pipelines predicting c
    """
    N, P = preds.shape
    pe = np.zeros(N)
    for i in range(N):
        counts = np.bincount(preds[i], minlength=n_classes).astype(float)
        q = counts / P
        q = q[q > 0]
        pe[i] = -np.sum(q * np.log(q + 1e-10))
    return pe


def compute_softmax_entropy(logits):
    """Compute softmax entropy from logits.

    H_soft(x_i) = -sum_c sigma(z)_c log(sigma(z)_c)
    """
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def compute_mc_uncertainty(mc_preds, n_classes):
    """Compute MC Dropout uncertainty (prediction disagreement)."""
    N, M = mc_preds.shape
    mc_unc = np.zeros(N)
    for i in range(N):
        counts = np.bincount(mc_preds[i], minlength=n_classes)
        mc_unc[i] = 1.0 - counts.max() / M
    return mc_unc


def abstention_curve(uncertainty, correct, coverages=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]):
    """Compute accuracy at different coverage levels (abstain on high-uncertainty trials)."""
    sorted_idx = np.argsort(uncertainty)  # lowest uncertainty first
    results = {}
    N = len(uncertainty)
    for cov in coverages:
        k = max(1, int(N * cov))
        keep = sorted_idx[:k]
        acc = correct[keep].mean()
        results[f"cov_{int(cov*100)}"] = float(acc)
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # Get subjects
    subject_dirs = sorted(data_dir.glob("S*"))
    all_subjects = [int(d.name[1:]) for d in subject_dirs]
    splits = create_subject_splits(len(all_subjects), n_folds=3, seed=42)

    all_results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        test_subjs = [all_subjects[i] for i in test_idx]
        logger.info(f"Fold {fold_idx}: test subjects {test_subjs}")

        test_ds = MultiPipelineDataset(data_dir, test_subjs, n_pipelines=128)
        sample, _ = test_ds[0]
        n_ch, n_t = sample.shape[1], sample.shape[2]
        n_classes = len(np.unique(test_ds.labels))

        # Load trained model
        model_path = model_dir / f"best_fold{fold_idx}.pt"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        model = get_model("eegnet", n_channels=n_ch, n_times=n_t,
                          n_classes=n_classes).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True,
                                          map_location=device))

        # Compute all predictions
        logger.info(f"  Running inference ({len(test_ds)} trials × 128 pipelines + MC Dropout)...")
        preds, logits_p0, mc_preds, labels = compute_trial_predictions(
            model, test_ds, device, n_mc_dropout=args.n_mc_dropout)

        # Compute uncertainty metrics
        pu = compute_pu(preds, n_classes)
        pe = compute_pe(preds, n_classes)
        h_soft = compute_softmax_entropy(logits_p0)
        mc_unc = compute_mc_uncertainty(mc_preds, n_classes)

        # Combined: normalized PU + softmax (simple average)
        pu_norm = (pu - pu.min()) / (pu.max() - pu.min() + 1e-8)
        h_norm = (h_soft - h_soft.min()) / (h_soft.max() - h_soft.min() + 1e-8)
        combined = 0.5 * pu_norm + 0.5 * h_norm

        # Correctness (pipeline 0)
        correct = (preds[:, 0] == labels).astype(float)
        is_error = 1 - correct

        # Correlations
        from scipy.stats import spearmanr
        rho_pu_soft, _ = spearmanr(pu, h_soft)
        rho_pu_mc, _ = spearmanr(pu, mc_unc)
        rho_soft_mc, _ = spearmanr(h_soft, mc_unc)

        # AUROC for error detection
        auroc_pu = roc_auc_score(is_error, pu) if len(np.unique(is_error)) > 1 else 0.5
        auroc_soft = roc_auc_score(is_error, h_soft) if len(np.unique(is_error)) > 1 else 0.5
        auroc_mc = roc_auc_score(is_error, mc_unc) if len(np.unique(is_error)) > 1 else 0.5
        auroc_comb = roc_auc_score(is_error, combined) if len(np.unique(is_error)) > 1 else 0.5

        # Abstention curves
        abs_pu = abstention_curve(pu, correct)
        abs_soft = abstention_curve(h_soft, correct)
        abs_mc = abstention_curve(mc_unc, correct)
        abs_comb = abstention_curve(combined, correct)

        fold_result = {
            "fold": fold_idx,
            "n_trials": len(labels),
            "baseline_acc": float(correct.mean()),
            "correlations": {
                "pu_vs_softmax": float(rho_pu_soft),
                "pu_vs_mc_dropout": float(rho_pu_mc),
                "softmax_vs_mc_dropout": float(rho_soft_mc),
            },
            "auroc_error_detection": {
                "PU": float(auroc_pu),
                "softmax": float(auroc_soft),
                "mc_dropout": float(auroc_mc),
                "PU_plus_softmax": float(auroc_comb),
            },
            "abstention": {
                "PU": abs_pu,
                "softmax": abs_soft,
                "mc_dropout": abs_mc,
                "PU_plus_softmax": abs_comb,
            },
            "pu_stats": {
                "mean": float(pu.mean()),
                "std": float(pu.std()),
                "median": float(np.median(pu)),
            },
        }
        all_results.append(fold_result)

        logger.info(f"  Fold {fold_idx}:")
        logger.info(f"    Correlations: PU-softmax={rho_pu_soft:.3f}, PU-MC={rho_pu_mc:.3f}")
        logger.info(f"    AUROC: PU={auroc_pu:.3f}, softmax={auroc_soft:.3f}, MC={auroc_mc:.3f}, combined={auroc_comb:.3f}")
        logger.info(f"    Acc@80%cov: PU={abs_pu['cov_80']:.3f}, softmax={abs_soft['cov_80']:.3f}, combined={abs_comb['cov_80']:.3f}")

    # Save results
    with open(out_dir / "pu_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {out_dir / 'pu_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="results_v4/pu")
    parser.add_argument("--n-mc-dropout", type=int, default=30)
    args = parser.parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    main(args)
