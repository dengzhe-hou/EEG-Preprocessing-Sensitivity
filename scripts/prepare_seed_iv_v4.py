#!/usr/bin/env python
"""
Prepare SEED-IV emotion dataset with 128 pipeline variants.
Data from BCMI (Shanghai Jiao Tong University).
4-class emotion: neutral(0), sad(1), fear(2), happy(3).
62 EEG channels, 200Hz, 15 subjects, 3 sessions, 24 trials/session.

Output: data/processed_v4/seed_iv/S{id}/all_pipelines.npy + labels.npy
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import build_pipeline_configs, N_PIPELINES
from src.preprocessing import robust_zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# SEED-IV emotion labels per session (24 trials each)
# From SEED-IV documentation
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

EPOCH_LEN = 4  # seconds


def main(args):
    out_dir = Path(args.output_dir) / "seed_iv"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_dir) / "SEED_IV" / "eeg_raw_data"
    if not raw_dir.exists():
        logger.error(f"Raw data not found at {raw_dir}")
        return

    configs = build_pipeline_configs()
    sfreq = 200  # SEED-IV preprocessed data is at 200Hz

    # Process each subject (use session 1 only for simplicity, or all 3)
    for subj_id in range(1, args.n_subjects + 1):
        subj_dir = out_dir / f"S{subj_id:02d}"
        subj_dir.mkdir(exist_ok=True)

        if (subj_dir / "all_pipelines.npy").exists() and not args.force:
            logger.info(f"S{subj_id:02d} already done, skipping.")
            continue

        logger.info(f"Processing subject {subj_id}...")

        all_epochs = []
        all_labels = []

        for session in range(1, 4):
            session_dir = raw_dir / str(session)
            # Find the .mat file for this subject in this session
            mat_files = list(session_dir.glob(f"{subj_id}_*.mat"))
            if not mat_files:
                logger.warning(f"  No .mat for subject {subj_id} session {session}")
                continue

            mat_path = mat_files[0]
            try:
                mat = sio.loadmat(str(mat_path))
            except Exception as e:
                logger.warning(f"  Failed to load {mat_path}: {e}")
                continue

            labels = SESSION_LABELS[session]

            # Each trial is stored as {prefix}_eeg{i} with shape (62, T)
            # Prefix varies per subject (e.g., cz_eeg, ha_eeg, hql_eeg)
            eeg_keys = sorted([k for k in mat.keys() if '_eeg' in k and not k.startswith('_')],
                             key=lambda k: int(k.split('_eeg')[1]))
            if not eeg_keys:
                logger.warning(f"  No EEG keys in {mat_path.name}")
                continue

            for trial_idx, key in enumerate(eeg_keys[:24]):
                trial_data = mat[key]  # (62, T)
                n_ch, n_times = trial_data.shape

                # Cut into non-overlapping EPOCH_LEN-second epochs
                samples_per_epoch = int(EPOCH_LEN * sfreq)
                n_epochs = n_times // samples_per_epoch

                for ep in range(n_epochs):
                    start = ep * samples_per_epoch
                    end = start + samples_per_epoch
                    epoch = trial_data[:, start:end]  # (62, 800)
                    all_epochs.append(epoch)
                    all_labels.append(labels[trial_idx])

        if len(all_epochs) < 50:
            logger.warning(f"  Too few epochs ({len(all_epochs)}), skipping.")
            continue

        base_data = np.stack(all_epochs).astype(np.float64)  # (N, 62, 800)
        labels = np.array(all_labels, dtype=np.int64)

        # Subsample to max 500 epochs (128 pipelines × 62ch × 800t × 500 = 12.8GB)
        if len(base_data) > 500:
            idx = np.random.choice(len(base_data), 500, replace=False)
            idx.sort()
            base_data = base_data[idx]
            labels = labels[idx]
            logger.info(f"  Subsampled to {len(base_data)} epochs")
        n_trials, n_ch, n_times = base_data.shape
        logger.info(f"  {n_trials} epochs, {n_ch} channels, {n_times} times, "
                    f"classes: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # Apply 128 pipeline variants
        all_pipelines = []
        min_trials = n_trials

        for config in configs:
            data = base_data.copy()

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
        np.save(subj_dir / "labels.npy", labels[:min_trials])
        logger.info(f"  Saved: {stacked.shape} ({stacked.nbytes/1e9:.1f} GB), {min_trials} trials")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed_v4")
    parser.add_argument("--raw-dir", default="raw/seed_iv")
    parser.add_argument("--n-subjects", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    np.random.seed(42)
    main(parser.parse_args())
