#!/usr/bin/env python
"""
Prepare CHB-MIT seizure dataset with 128 pipeline variants.
Auto-downloads from PhysioNet. Binary: seizure(1) vs non-seizure(0).
23 channels, 256Hz. Use 10 subjects for pilot.
Output: data/processed_v4/chbmit/S{id}/all_pipelines.npy + labels.npy
"""
import argparse, logging, sys, os
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
import mne

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import build_pipeline_configs, N_PIPELINES
from src.preprocessing import robust_zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")

# CHB-MIT common channels (18 channels present across most subjects)
COMMON_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]


def download_chbmit(data_dir, subjects):
    """Download CHB-MIT data from PhysioNet using MNE."""
    import pooch
    base_url = "https://physionet.org/files/chbmit/1.0.0/"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        subj_str = f"chb{subj:02d}"
        subj_dir = data_dir / subj_str
        if subj_dir.exists() and len(list(subj_dir.glob("*.edf"))) > 0:
            logger.info(f"  {subj_str} already downloaded")
            continue

        subj_dir.mkdir(exist_ok=True)
        # Download summary file
        summary_url = f"{base_url}{subj_str}/{subj_str}-summary.txt"
        try:
            summary_path = pooch.retrieve(summary_url, known_hash=None,
                                           path=str(subj_dir), fname=f"{subj_str}-summary.txt")
        except Exception as e:
            logger.warning(f"  Failed to download summary for {subj_str}: {e}")
            continue

        # Parse summary to get file list and seizure times
        with open(summary_path) as f:
            content = f.read()

        # Download EDF files mentioned in summary
        import re
        edf_files = re.findall(r'File Name: (\S+\.edf)', content)
        for edf in edf_files[:5]:  # limit to 5 files per subject for speed
            edf_url = f"{base_url}{subj_str}/{edf}"
            try:
                pooch.retrieve(edf_url, known_hash=None,
                              path=str(subj_dir), fname=edf)
            except Exception as e:
                logger.warning(f"  Failed to download {edf}: {e}")


def parse_seizure_times(summary_path):
    """Parse seizure start/end times from CHB-MIT summary file."""
    seizures = {}  # file -> [(start, end), ...]
    current_file = None

    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
                seizures[current_file] = []
            elif line.startswith("Seizure Start Time:") or line.startswith("Seizure  Start Time:"):
                start = int(line.split(":")[1].strip().split()[0])
                seizures.setdefault(current_file, [])
            elif line.startswith("Seizure End Time:") or line.startswith("Seizure  End Time:"):
                end = int(line.split(":")[1].strip().split()[0])
                if current_file and seizures.get(current_file) is not None:
                    seizures[current_file].append((start, end))

    return seizures


def main(args):
    out_dir = Path(args.output_dir) / "chbmit"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    configs = build_pipeline_configs()
    subjects = list(range(1, args.n_subjects + 1))

    # Download data
    logger.info("Downloading CHB-MIT data...")
    download_chbmit(raw_dir, subjects)

    for subj in subjects:
        subj_str = f"chb{subj:02d}"
        subj_out = out_dir / f"S{subj:02d}"
        subj_out.mkdir(exist_ok=True)

        if (subj_out / "all_pipelines.npy").exists() and not args.force:
            logger.info(f"S{subj:02d} already done, skipping.")
            continue

        logger.info(f"Processing subject {subj}...")
        subj_raw = raw_dir / subj_str

        # Parse seizure times
        summary_f = subj_raw / f"{subj_str}-summary.txt"
        if not summary_f.exists():
            logger.warning(f"  No summary file for {subj_str}")
            continue
        seizures = parse_seizure_times(summary_f)

        # Load EDF files and create epochs
        all_epochs = []
        all_labels = []
        epoch_len = 4  # seconds

        for edf_file in sorted(subj_raw.glob("*.edf")):
            try:
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            except Exception as e:
                continue

            sfreq = raw.info['sfreq']
            duration = raw.times[-1]

            # Find common channels
            available = [ch for ch in COMMON_CHANNELS if ch in raw.ch_names]
            if len(available) < 10:
                continue
            raw.pick(available)
            n_ch = len(available)

            # Get seizure intervals for this file
            fname = edf_file.name
            file_seizures = seizures.get(fname, [])

            # Create non-overlapping 4s epochs using sample-based slicing
            full_data = raw.get_data()
            samples_per_epoch = int(epoch_len * sfreq)
            n_epochs = full_data.shape[1] // samples_per_epoch
            for i in range(n_epochs):
                start_s = i * samples_per_epoch
                end_s = start_s + samples_per_epoch
                start_sec = i * epoch_len
                end_sec = start_sec + epoch_len
                is_seizure = any(s <= end_sec and e >= start_sec for s, e in file_seizures)

                epoch_data = full_data[:, start_s:end_s]
                if epoch_data.shape[1] == samples_per_epoch:
                    all_epochs.append(epoch_data)
                    all_labels.append(1 if is_seizure else 0)

        if len(all_epochs) < 50:
            logger.warning(f"  Too few epochs ({len(all_epochs)}), skipping.")
            continue

        base_data = np.stack(all_epochs)  # (N, C, T)
        labels = np.array(all_labels, dtype=np.int64)
        n_trials, n_ch, n_times = base_data.shape
        sfreq_actual = n_times / epoch_len

        # Balance: subsample non-seizure to 5:1 ratio
        seizure_idx = np.where(labels == 1)[0]
        nonseizure_idx = np.where(labels == 0)[0]
        if len(seizure_idx) > 0 and len(nonseizure_idx) > 5 * len(seizure_idx):
            keep_ns = np.random.choice(nonseizure_idx, 5 * len(seizure_idx), replace=False)
            keep = np.sort(np.concatenate([seizure_idx, keep_ns]))
            base_data = base_data[keep]
            labels = labels[keep]
            n_trials = len(labels)

        logger.info(f"  {n_trials} epochs ({labels.sum()} seizure), {n_ch} channels, sfreq={sfreq_actual:.0f}")

        # Apply 128 pipeline variants
        all_pipelines = []
        min_trials = n_trials

        for config in configs:
            data = base_data.copy()

            if config["reference"] == "car":
                data = data - data.mean(axis=1, keepdims=True)

            if config["hpf"] == "0.5hz":
                sos = butter(4, 0.5, btype='high', fs=sfreq_actual, output='sos')
                for t in range(n_trials):
                    for c in range(n_ch):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            if config["lpf"] == "30hz":
                sos = butter(4, 30, btype='low', fs=sfreq_actual, output='sos')
                for t in range(n_trials):
                    for c in range(n_ch):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            if config["baseline"] == "200ms":
                bl = int(0.2 * sfreq_actual)
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

        if min_trials < 20:
            logger.warning(f"  Too few trials after filtering ({min_trials}), skipping.")
            continue

        stacked = np.stack([p[:min_trials] for p in all_pipelines], axis=0)
        np.save(subj_out / "all_pipelines.npy", stacked)
        np.save(subj_out / "labels.npy", labels[:min_trials])
        logger.info(f"  Saved: {stacked.shape} ({stacked.nbytes/1e9:.1f} GB), {min_trials} trials")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed_v4")
    parser.add_argument("--raw-dir", default="data/raw/chbmit")
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    np.random.seed(42)
    main(parser.parse_args())
