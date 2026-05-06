#!/usr/bin/env python
"""
Prepare Sleep-EDF Expanded with 128 pipeline variants (v4).

Auto-downloads from PhysioNet via MNE.
5-class sleep staging: W(0), N1(1), N2(2), N3(3), REM(4).
30-second epochs, 2 EEG channels.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervention_graph import build_pipeline_configs, N_PIPELINES
from src.preprocessing import robust_zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")

STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}


def main(args):
    out_dir = Path(args.output_dir) / "sleep_edf"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = build_pipeline_configs()

    for subj in range(args.n_subjects):
        subj_dir = out_dir / f"S{subj:02d}"
        subj_dir.mkdir(exist_ok=True)

        if (subj_dir / "all_pipelines.npy").exists() and not args.force:
            logger.info(f"S{subj:02d} already done, skipping.")
            continue

        logger.info(f"Processing subject {subj}...")
        try:
            result = fetch_data(subjects=[subj], recording=[1])
            psg_file, hyp_file = result[0]  # unpack nested list
        except Exception as e:
            logger.warning(f"  Failed to download subject {subj}: {e}")
            continue

        try:
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            annotations = mne.read_annotations(hyp_file)
            raw.set_annotations(annotations, verbose=False)
        except Exception as e:
            logger.warning(f"  Failed to load subject {subj}: {e}")
            continue

        # Keep EEG channels only
        eeg_chs = [ch for ch in raw.ch_names if 'EEG' in ch]
        if len(eeg_chs) < 2:
            eeg_chs = raw.ch_names[:2]
        raw.pick(eeg_chs[:2])  # exactly 2 channels
        sfreq = raw.info['sfreq']  # typically 100 Hz

        # Create 30s epochs from annotations
        events, event_id = mne.events_from_annotations(
            raw, chunk_duration=30.0, verbose=False)

        # Map event codes to sleep stages
        inv_map = {v: k for k, v in event_id.items()}
        keep_events = []
        labels = []
        for ev in events:
            desc = inv_map.get(ev[2], "")
            stage = STAGE_MAP.get(desc, -1)
            if stage >= 0:
                keep_events.append(ev)
                labels.append(stage)

        if len(keep_events) < 50:
            logger.warning(f"  Too few epochs ({len(keep_events)}), skipping.")
            continue

        keep_events = np.array(keep_events)
        labels = np.array(labels, dtype=np.int64)

        # Create base epochs (no preprocessing yet — just epoching)
        epochs = mne.Epochs(raw, keep_events, tmin=0,
                           tmax=30.0 - 1.0/sfreq,
                           baseline=None, preload=True, verbose=False)
        base_data = epochs.get_data(copy=True)  # (N, 2, T)
        # Truncate labels to match
        n_ep = min(len(base_data), len(labels))
        base_data = base_data[:n_ep]
        labels = labels[:n_ep]

        n_trials, n_ch, n_times = base_data.shape
        logger.info(f"  {n_trials} epochs, {n_ch} channels, {n_times} times, sfreq={sfreq}")

        # Apply 128 pipeline variants
        all_pipelines = []
        min_trials = n_trials

        for config in configs:
            data = base_data.copy()

            # a1: reference
            if config["reference"] == "car":
                data = data - data.mean(axis=1, keepdims=True)

            # a2: hpf (0.1→0.5)
            if config["hpf"] == "0.5hz":
                sos = butter(4, 0.5, btype='high', fs=sfreq, output='sos')
                for t in range(n_trials):
                    for c in range(n_ch):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            # a3: lpf (45→30)
            if config["lpf"] == "30hz":
                sos = butter(4, 30, btype='low', fs=sfreq, output='sos')
                for t in range(n_trials):
                    for c in range(n_ch):
                        data[t, c] = sosfiltfilt(sos, data[t, c])

            # a4: baseline
            if config["baseline"] == "200ms":
                bl_samp = int(0.2 * sfreq)
                if 0 < bl_samp < n_times:
                    data = data - data[:, :, :bl_samp].mean(axis=2, keepdims=True)

            # a5: asr
            if config["asr"] == "on":
                thr = 20 * np.median(np.abs(data))
                data = np.clip(data, -thr, thr)

            # a6: autoreject
            keep = np.ones(n_trials, dtype=bool)
            if config["autoreject"] == "interp":
                mx = np.abs(data).max(axis=(1, 2))
                med = np.median(mx)
                mad = np.median(np.abs(mx - med)) * 1.4826
                keep = mx < (med + 3 * mad)

            # a7: bad channel
            if config["badchannel"] == "ransac" and n_ch > 1:
                ch_var = data.var(axis=(0, 2))
                med_var = np.median(ch_var)
                bad = ch_var > 5 * med_var
                if bad.any() and not bad.all():
                    data[:, bad, :] = data[:, ~bad, :].mean(axis=1, keepdims=True)

            data = data[keep]
            data = robust_zscore(data, axis=-1).astype(np.float32)
            all_pipelines.append(data)
            min_trials = min(min_trials, len(data))

        # Truncate all to common trial count and stack
        stacked = np.stack([p[:min_trials] for p in all_pipelines], axis=0)  # (128, N, 2, T)
        np.save(subj_dir / "all_pipelines.npy", stacked)
        np.save(subj_dir / "labels.npy", labels[:min_trials])
        logger.info(f"  Saved: {stacked.shape} ({stacked.nbytes/1e9:.1f} GB), {min_trials} trials")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed_v4")
    parser.add_argument("--n-subjects", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
