#!/usr/bin/env python
"""
Prepare PhysionetMI (EEGBCI) with 128 pipeline variants.
Auto-downloads via MOABB. 2-class MI (hands vs feet), 64 channels, 160Hz.
Output: data/processed_v4/physionet_mi/S{id}/all_pipelines.npy + labels.npy
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import build_pipeline_configs, N_PIPELINES
from src.preprocessing import robust_zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    out_dir = Path(args.output_dir) / "physionet_mi"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PhysionetMI()
    paradigm = MotorImagery(n_classes=2, fmin=0.1, fmax=45, resample=160)
    configs = build_pipeline_configs()

    subjects = dataset.subject_list[:args.n_subjects]
    logger.info(f"Using {len(subjects)} of {len(dataset.subject_list)} subjects")

    for subj in subjects:
        subj_dir = out_dir / f"S{subj:02d}"
        subj_dir.mkdir(exist_ok=True)

        if (subj_dir / "all_pipelines.npy").exists() and not args.force:
            logger.info(f"S{subj:02d} already done, skipping.")
            continue

        logger.info(f"Processing subject {subj}...")
        try:
            X, labels, meta = paradigm.get_data(dataset, [subj])
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            continue

        unique = sorted(set(labels))
        label_map = {l: i for i, l in enumerate(unique)}
        int_labels = np.array([label_map[l] for l in labels])

        n_trials, n_ch, n_times = X.shape
        sfreq = 160
        logger.info(f"  Shape: {X.shape}, Classes: {unique}, sfreq={sfreq}")

        all_pipelines = []
        min_trials = n_trials

        for config in configs:
            data = X.copy()

            if config["reference"] == "car":
                data = data - data.mean(axis=1, keepdims=True)

            if config["hpf"] == "0.5hz":
                sos = butter(4, 0.5, btype='high', fs=sfreq, output='sos')
                for t in range(n_trials):
                    for c in range(n_ch):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            if config["lpf"] == "30hz":
                sos = butter(4, 30, btype='low', fs=sfreq, output='sos')
                for t in range(n_trials):
                    for c in range(n_ch):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            if config["baseline"] == "200ms":
                bl = int(0.2 * sfreq)
                if 0 < bl < n_times:
                    data = data - data[:, :, :bl].mean(axis=2, keepdims=True)

            if config["asr"] == "on":
                thr = 20 * np.median(np.abs(data))
                data = np.clip(data, -thr, thr)

            keep = np.ones(n_trials, dtype=bool)
            if config["autoreject"] == "interp":
                mx = np.abs(data).max(axis=(1, 2))
                med = np.median(mx)
                mad = np.median(np.abs(mx - med)) * 1.4826
                keep = mx < (med + 3 * mad)

            if config["badchannel"] == "ransac":
                ch_var = data.var(axis=(0, 2))
                med_var = np.median(ch_var)
                bad = (ch_var > 5 * med_var) | (ch_var < 0.1 * med_var)
                if bad.any() and not bad.all():
                    data[:, bad, :] = data[:, ~bad, :].mean(axis=1, keepdims=True)

            data = data[keep]
            data = robust_zscore(data, axis=-1).astype(np.float32)
            all_pipelines.append(data)
            min_trials = min(min_trials, len(data))

        stacked = np.stack([p[:min_trials] for p in all_pipelines], axis=0)
        np.save(subj_dir / "all_pipelines.npy", stacked)
        np.save(subj_dir / "labels.npy", int_labels[:min_trials])
        logger.info(f"  Saved: {stacked.shape} ({stacked.nbytes/1e9:.1f} GB), {min_trials} trials")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed_v4")
    parser.add_argument("--n-subjects", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
