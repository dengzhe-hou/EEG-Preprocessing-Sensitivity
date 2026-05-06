"""
Multi-pipeline EEG dataset with memory-mapped loading.

v4: Uses merged per-subject files (all_pipelines.npy, shape 128×N×C×T)
    with numpy memory mapping for fast random access without loading into RAM.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MultiPipelineDataset(Dataset):
    """Memory-mapped dataset for 128-pipeline experiments.

    Each subject has:
      - all_pipelines.npy: shape (128, N_trials, C, T) — memory-mapped
      - labels.npy: shape (N_trials,)

    Args:
        data_dir: path to dataset root
        subjects: list of subject IDs
        n_pipelines: total number of pipelines (default 128)
        pipeline_indices: subset to use (default all)
        normalize: per-channel z-scoring on the fly
        n_views_sample: randomly sample this many views per __getitem__ (None=all)
    """

    def __init__(self, data_dir, subjects, n_pipelines=128,
                 pipeline_indices=None, normalize=True, n_views_sample=None):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.n_views_sample = n_views_sample

        if pipeline_indices is None:
            self.pipeline_indices = list(range(n_pipelines))
        else:
            self.pipeline_indices = sorted(pipeline_indices)
        self.n_views = len(self.pipeline_indices)

        # Memory-map all subject files
        self.mmaps = []     # list of (mmap_array, offset) — mmap shape: (128, N, C, T)
        self.labels_list = []
        self.index = []     # (mmap_idx, trial_idx) per sample

        for subj in subjects:
            subj_dir = self.data_dir / f"S{subj:02d}"
            merged_f = subj_dir / "all_pipelines.npy"
            labels_f = subj_dir / "labels.npy"

            if not merged_f.exists() or not labels_f.exists():
                continue

            mmap = np.load(str(merged_f), mmap_mode='r')  # (128, N, C, T)
            labels = np.load(str(labels_f))
            n_trials = min(mmap.shape[1], len(labels))

            mmap_idx = len(self.mmaps)
            self.mmaps.append(mmap)
            self.labels_list.append(labels[:n_trials])

            for t in range(n_trials):
                self.index.append((mmap_idx, t))

        self.all_labels = np.concatenate(self.labels_list) if self.labels_list else np.array([])

        # Detect dimensions
        if self.mmaps:
            self.n_channels = self.mmaps[0].shape[2]
            self.n_times = self.mmaps[0].shape[3]
        else:
            self.n_channels = 22
            self.n_times = 1001

    @property
    def labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        mmap_idx, trial_idx = self.index[idx]
        mmap = self.mmaps[mmap_idx]
        label = self.labels_list[mmap_idx][trial_idx]

        # Select views
        if self.n_views_sample is not None and self.n_views_sample < self.n_views:
            perm = np.random.permutation(self.n_views)[:self.n_views_sample]
            selected = [self.pipeline_indices[p] for p in perm]
        else:
            selected = self.pipeline_indices

        # Read from memory-mapped array (fast — OS page cache)
        views = mmap[selected, trial_idx, :, :].astype(np.float32)  # (V, C, T)

        if self.normalize:
            mean = views.mean(axis=-1, keepdims=True)
            std = views.std(axis=-1, keepdims=True)
            std = np.maximum(std, 1e-8)
            views = (views - mean) / std

        return torch.from_numpy(views.copy()), torch.tensor(label, dtype=torch.long)


def create_subject_splits(n_subjects, n_folds=5, seed=42):
    """Create leave-group-out CV splits by subject."""
    rng = np.random.RandomState(seed)
    subjects = np.arange(n_subjects)
    rng.shuffle(subjects)

    # Distribute subjects evenly across folds, giving remainder to earlier folds
    fold_indices = np.array_split(subjects, n_folds)
    splits = []
    for i in range(n_folds):
        test = fold_indices[i]
        remaining = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
        val_size = max(1, len(remaining) // 4)
        val = remaining[:val_size]
        train = remaining[val_size:]
        splits.append((train.tolist(), val.tolist(), test.tolist()))

    return splits
