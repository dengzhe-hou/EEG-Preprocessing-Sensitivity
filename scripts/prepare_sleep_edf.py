#!/usr/bin/env python
"""
Prepare Sleep-EDF Expanded dataset with 128 pipeline variants.

Uses mne.datasets.sleep_physionet (auto-downloads from PhysioNet).
5-class sleep staging: W(0), N1(1), N2(2), N3(3), REM(4).
30-second epochs, 2 EEG channels (Fpz-Cz, Pz-Oz).

Output: data/processed_v4/sleep_edf/S{id}/pipeline_{000-127}.npz + labels.npy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import build_pipeline_configs, N_PIPELINES
from src.preprocessing import robust_zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")

# Sleep stage mapping
ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # merge N3+N4
    "Sleep stage R": 4,
}


def main(args):
    out_dir = Path(args.output_dir) / "sleep_edf"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_pipeline_configs()
    subjects = list(range(args.n_subjects))

    for subj in subjects:
        logger.info(f"Processing subject {subj}...")
        subj_dir = out_dir / f"S{subj:02d}"
        subj_dir.mkdir(exist_ok=True)

        if (subj_dir / "labels.npy").exists() and not args.force:
            logger.info(f"  Subject {subj} already done, skipping.")
            continue

        try:
            [psg_file, hyp_file] = fetch_data(subjects=[subj], recording=[1])
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            annotations = mne.read_annotations(hyp_file)
            raw.set_annotations(annotations, verbose=False)
        except Exception as e:
            logger.warning(f"  Failed to load subject {subj}: {e}")
            continue

        # Keep only EEG channels
        eeg_channels = [ch for ch in raw.ch_names
                        if 'EEG' in ch or 'Fpz' in ch or 'Pz' in ch]
        if not eeg_channels:
            eeg_channels = raw.ch_names[:2]  # fallback to first 2
        raw.pick(eeg_channels)

        # Create events from annotations
        events, event_id_raw = mne.events_from_annotations(
            raw, chunk_duration=30.0, verbose=False)

        # Map to our 5-class scheme
        event_id = {}
        for desc, eid in event_id_raw.items():
            if desc in ANNOTATION_MAP:
                event_id[desc] = eid

        if not event_id:
            logger.warning(f"  No valid sleep stages for subject {subj}")
            continue

        # Create base epochs
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=30.0 - 1.0/raw.info['sfreq'],
                           baseline=None, preload=True, verbose=False)

        base_data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
        base_labels = np.array([ANNOTATION_MAP.get(epochs.events[i, -1], -1)
                                for i in range(len(epochs))])

        # Remap event codes to our labels
        inv_map = {v: k for k, v in event_id.items()}
        base_labels = []
        for ev in epochs.events[:, -1]:
            desc = inv_map.get(ev, "")
            base_labels.append(ANNOTATION_MAP.get(desc, -1))
        base_labels = np.array(base_labels)

        # Filter out unknown stages
        valid = base_labels >= 0
        base_data = base_data[valid]
        base_labels = base_labels[valid]

        if len(base_data) == 0:
            logger.warning(f"  No valid epochs for subject {subj}")
            continue

        n_trials, n_channels, n_times = base_data.shape
        sfreq = raw.info['sfreq']
        logger.info(f"  Shape: {base_data.shape}, sfreq: {sfreq}")

        for config in configs:
            idx = config["index"]
            data = base_data.copy()

            # a1: reference (with only 2 channels, CAR = flip polarity — still valid as intervention)
            if config["reference"] == "car":
                data = data - data.mean(axis=1, keepdims=True)

            # a2: hpf
            if config["hpf"] == "0.5hz":
                from scipy.signal import butter, sosfiltfilt
                sos = butter(4, 0.5, btype='high', fs=sfreq, output='sos')
                for t in range(n_trials):
                    for c in range(n_channels):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            # a3: lpf
            if config["lpf"] == "30hz":
                from scipy.signal import butter, sosfiltfilt
                sos = butter(4, 30, btype='low', fs=sfreq, output='sos')
                for t in range(n_trials):
                    for c in range(n_channels):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            # a4: baseline
            if config["baseline"] == "200ms":
                bl_samp = int(0.2 * sfreq)
                if bl_samp > 0 and bl_samp < n_times:
                    bl = data[:, :, :bl_samp].mean(axis=2, keepdims=True)
                    data = data - bl

            # a5: ASR
            if config["asr"] == "on":
                threshold = 20 * np.median(np.abs(data))
                data = np.clip(data, -threshold, threshold)

            # a6: autoreject
            keep_mask = np.ones(n_trials, dtype=bool)
            if config["autoreject"] == "interp":
                epoch_max = np.abs(data).max(axis=(1, 2))
                med = np.median(epoch_max)
                mad = np.median(np.abs(epoch_max - med)) * 1.4826
                keep_mask = epoch_max < (med + 3 * mad)

            # a7: bad channel (with 2 channels, mostly a no-op but keeps structure)
            if config["badchannel"] == "ransac":
                ch_var = data.var(axis=(0, 2))
                med_var = np.median(ch_var)
                bad = ch_var > 5 * med_var
                if bad.any() and not bad.all():
                    good_mean = data[:, ~bad, :].mean(axis=1, keepdims=True)
                    data[:, bad, :] = good_mean

            data = data[keep_mask]
            data = robust_zscore(data, axis=-1).astype(np.float32)
            np.savez_compressed(subj_dir / f"pipeline_{idx:03d}.npz", data=data)

        # Truncate to common trial count
        min_trials = min(
            np.load(subj_dir / f"pipeline_{i:03d}.npz")["data"].shape[0]
            for i in range(N_PIPELINES)
            if (subj_dir / f"pipeline_{i:03d}.npz").exists()
        )
        for i in range(N_PIPELINES):
            f = subj_dir / f"pipeline_{i:03d}.npz"
            if f.exists():
                d = np.load(f)["data"][:min_trials]
                np.savez_compressed(f, data=d)
        np.save(subj_dir / "labels.npy", base_labels[:min_trials])
        logger.info(f"  Saved {N_PIPELINES} pipelines, {min_trials} trials for S{subj}")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed_v4")
    parser.add_argument("--n-subjects", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
