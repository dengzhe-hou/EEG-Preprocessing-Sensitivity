#!/usr/bin/env python
"""
Download and preprocess BCI Competition IV-2a dataset using MOABB.

Generates 16 pipeline variants and saves as numpy arrays.
Output: data/processed/bci_iv_2a/{subject_id}/{pipeline_idx}.npz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import apply_pipeline, robust_zscore
from src.intervention_graph import build_pipeline_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    out_dir = Path(args.output_dir) / "bci_iv_2a"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=4, fmax=38, tmin=0, tmax=4, resample=250
    )

    configs = build_pipeline_configs()
    subjects = dataset.subject_list

    for subj in subjects:
        logger.info(f"Processing subject {subj}...")
        subj_dir = out_dir / f"S{subj:02d}"
        subj_dir.mkdir(exist_ok=True)

        # Check if already done
        if (subj_dir / "labels.npy").exists() and not args.force:
            logger.info(f"  Subject {subj} already processed, skipping.")
            continue

        try:
            X, labels, meta = paradigm.get_data(dataset, [subj])
        except Exception as e:
            logger.warning(f"  Failed to load subject {subj}: {e}")
            continue

        # Save labels (same for all pipelines)
        # Convert string labels to integers
        unique_labels = sorted(set(labels))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        int_labels = np.array([label_map[l] for l in labels])
        np.save(subj_dir / "labels.npy", int_labels)

        # X from MOABB is already (n_trials, n_channels, n_times)
        # This is the "base" preprocessed version (paradigm applies filter + resample)
        # For our experiment, we want to apply our 16 pipeline variants

        # Save the MOABB-processed version as pipeline 0 (baseline)
        # For simplicity in this pilot, we save the paradigm-processed data
        # as all 16 variants with slight modifications
        # In the full version, we'd work from raw data

        # For BCI-IV-2a via MOABB, the data is already epoched
        # We apply post-hoc variations where possible
        n_trials, n_channels, n_times = X.shape
        logger.info(f"  Shape: {X.shape}, Classes: {unique_labels}")

        for config in configs:
            idx = config["index"]
            data = X.copy()

            # Apply reference variation
            if config["reference"] == "car":
                mean_ref = data.mean(axis=1, keepdims=True)
                data = data - mean_ref

            # Apply normalization variation based on resample flag
            if config["resample"] == "128hz":
                # Subsample to roughly 128 Hz equivalent
                ratio = 128 / 250
                new_n = int(n_times * ratio)
                indices = np.linspace(0, n_times - 1, new_n).astype(int)
                data = data[:, :, indices]

            # Robust z-score
            data = robust_zscore(data, axis=-1).astype(np.float32)

            np.savez_compressed(
                subj_dir / f"pipeline_{idx:02d}.npz",
                data=data
            )

        logger.info(f"  Saved {len(configs)} pipelines for subject {subj}")

    # Save metadata
    meta_info = {
        "dataset": "BCI Competition IV-2a (BNCI2014-001)",
        "n_subjects": len(subjects),
        "n_classes": 4,
        "class_names": ["left_hand", "right_hand", "feet", "tongue"],
        "paradigm": "motor_imagery",
        "sfreq": 250,
        "n_channels": 22,
    }
    np.savez(out_dir / "meta.npz", **meta_info)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing")
    main(parser.parse_args())
