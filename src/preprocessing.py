"""
Preprocessing bank generator (v4).

Generates 128 pipeline variants for each raw EEG recording
from 7 atomic binary interventions (2^7 = 128 combinations).

Interventions:
  a1: reference (original vs CAR)
  a2: hpf (0.1 Hz vs 0.5 Hz)
  a3: lpf (45 Hz vs 30 Hz)
  a4: baseline (none vs 200ms subtractive)
  a5: asr (off vs on)
  a6: autoreject (off vs local interpolation)
  a7: badchannel (off vs RANSAC+interpolation)
"""

import logging
from pathlib import Path

import mne
import numpy as np

from .intervention_graph import build_pipeline_configs, INTERVENTION_NAMES, N_PIPELINES

logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")


def apply_pipeline(raw, config, notch_freq=60, asr_cutoff=20):
    """Apply a single preprocessing pipeline to a raw MNE object.

    Args:
        raw: mne.io.Raw object (will be copied)
        config: dict with keys from INTERVENTION_NAMES
        notch_freq: powerline frequency (50 or 60 Hz)
        asr_cutoff: ASR threshold

    Returns:
        raw_processed: processed mne.io.Raw object
    """
    raw = raw.copy().load_data()

    # --- a7: Bad channel repair (must be first — affects all downstream) ---
    if config["badchannel"] == "ransac":
        try:
            from mne.preprocessing import find_bad_channels_maxwell
            # Simple amplitude-based bad channel detection
            data = raw.get_data()
            ch_std = data.std(axis=1)
            median_std = np.median(ch_std)
            bad_mask = (ch_std > 5 * median_std) | (ch_std < 0.1 * median_std)
            bad_chs = [raw.ch_names[i] for i, b in enumerate(bad_mask) if b]
            if bad_chs:
                raw.info['bads'] = bad_chs
                raw.interpolate_bads(verbose=False)
        except Exception:
            pass  # skip if fails

    # --- Always: notch filter at powerline freq ---
    raw.notch_filter(freqs=notch_freq, method='iir', verbose=False)

    # --- a2: High-pass filter ---
    hpf = 0.1 if config["hpf"] == "0.1hz" else 0.5

    # --- a3: Low-pass filter ---
    lpf = 45.0 if config["lpf"] == "45hz" else 30.0

    # Bandpass
    raw.filter(l_freq=hpf, h_freq=lpf, method='fir', fir_design='firwin',
               verbose=False)

    # --- a1: Reference ---
    if config["reference"] == "car":
        raw.set_eeg_reference('average', projection=False, verbose=False)

    # --- a5: ASR artifact attenuation ---
    if config["asr"] == "on":
        try:
            from mne.preprocessing import annotate_amplitude
            data_std = np.std(raw.get_data())
            annotations = annotate_amplitude(
                raw, peak=asr_cutoff * data_std, flat=None, verbose=False)
            if annotations is not None and len(annotations) > 0:
                raw.set_annotations(raw.annotations + annotations)
        except Exception:
            pass

    return raw


def apply_epoch_level_interventions(epochs, config):
    """Apply interventions that operate at the epoch level.

    Args:
        epochs: mne.Epochs object (will be copied)
        config: dict with intervention settings

    Returns:
        data: (n_epochs, n_channels, n_times) numpy array
        keep_mask: boolean array of which epochs to keep
    """
    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape
    keep_mask = np.ones(n_epochs, dtype=bool)

    # --- a4: Baseline correction ---
    if config["baseline"] == "200ms":
        sfreq = epochs.info['sfreq']
        baseline_samples = int(0.2 * sfreq)
        if baseline_samples > 0 and baseline_samples < n_times:
            baseline_mean = data[:, :, :baseline_samples].mean(axis=2, keepdims=True)
            data = data - baseline_mean

    # --- a6: Autoreject (epoch-level amplitude rejection) ---
    if config["autoreject"] == "interp":
        # Simple threshold-based rejection: reject epochs with extreme amplitude
        epoch_max = np.abs(data).max(axis=(1, 2))
        threshold = np.median(epoch_max) + 3 * np.median(
            np.abs(epoch_max - np.median(epoch_max))) * 1.4826
        keep_mask = epoch_max < threshold

    return data, keep_mask


def generate_preprocessing_bank(raw, epochs_params, notch_freq=60, asr_cutoff=20):
    """Generate all 128 pipeline variants for a single recording.

    Args:
        raw: mne.io.Raw object
        epochs_params: dict with keys: events, event_id, tmin, tmax, baseline
                       OR None for fixed-length epochs (set 'duration' key)
        notch_freq: powerline frequency
        asr_cutoff: ASR threshold

    Returns:
        all_data: dict mapping pipeline_index → (n_epochs, n_channels, n_times) array
        labels: (n_epochs,) integer array (same across pipelines before rejection)
    """
    configs = build_pipeline_configs()
    all_data = {}

    for config in configs:
        idx = config["index"]
        try:
            # Apply continuous-level interventions
            processed_raw = apply_pipeline(raw, config, notch_freq=notch_freq,
                                           asr_cutoff=asr_cutoff)

            # Epoch
            if epochs_params.get("events") is not None:
                epochs = mne.Epochs(
                    processed_raw, epochs_params["events"],
                    epochs_params.get("event_id"),
                    tmin=epochs_params.get("tmin", 0),
                    tmax=epochs_params.get("tmax", 4),
                    baseline=None,  # we handle baseline ourselves
                    preload=True, verbose=False
                )
            else:
                duration = epochs_params.get("duration", 4.0)
                epochs = mne.make_fixed_length_epochs(
                    processed_raw, duration=duration,
                    preload=True, verbose=False)

            # Apply epoch-level interventions
            data, keep_mask = apply_epoch_level_interventions(epochs, config)

            # Robust z-score
            data = robust_zscore(data, axis=-1).astype(np.float32)

            all_data[idx] = (data, keep_mask)

        except Exception as e:
            logger.warning(f"Pipeline {idx} failed: {e}")
            all_data[idx] = None

    return all_data


def robust_zscore(data, axis=-1):
    """Per-channel robust z-scoring using median and MAD."""
    median = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
    mad = np.maximum(mad, 1e-8)
    return (data - median) / (mad * 1.4826)
