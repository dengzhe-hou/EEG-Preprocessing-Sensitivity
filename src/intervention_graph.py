"""
Intervention graph for the preprocessing semigroup.

K atomic binary interventions generate a Boolean lattice (K-dimensional hypercube)
with 2^K nodes and K·2^(K-1) undirected edges.

v4: 7 interventions (reference, hpf, lpf, baseline, asr, autoreject, badchannel)
    → 128 pipelines, 448 edges, diameter 7
"""

import itertools
from dataclasses import dataclass

import torch


@dataclass
class InterventionEdge:
    src: int          # source pipeline index
    dst: int          # destination pipeline index
    intervention: int # which atomic intervention (0..K-1)
    name: str         # human-readable name


# v4: 7 high-impact interventions (Kessler 2025, Del Pup 2025)
INTERVENTION_NAMES = [
    "reference",    # a1: native vs CAR
    "hpf",          # a2: 0.1 Hz vs 0.5 Hz high-pass
    "lpf",          # a3: 45 Hz vs 30 Hz low-pass
    "baseline",     # a4: none vs 200ms subtractive
    "asr",          # a5: off vs on (threshold=20)
    "autoreject",   # a6: off vs local interpolation
    "badchannel",   # a7: off vs RANSAC detection + interpolation
]

INTERVENTION_OPTIONS = [
    ("original", "car"),         # reference
    ("0.1hz", "0.5hz"),          # hpf
    ("45hz", "30hz"),            # lpf
    ("none", "200ms"),           # baseline
    ("off", "on"),               # asr
    ("off", "interp"),           # autoreject
    ("off", "ransac"),           # badchannel
]

K = len(INTERVENTION_NAMES)
N_PIPELINES = 2 ** K  # 128


def build_pipeline_configs():
    """Generate all 2^K pipeline configurations as binary tuples."""
    configs = []
    for bits in itertools.product(range(2), repeat=K):
        config = {INTERVENTION_NAMES[i]: INTERVENTION_OPTIONS[i][b]
                  for i, b in enumerate(bits)}
        config["index"] = sum(b << i for i, b in enumerate(bits))
        configs.append(config)
    return configs


def build_hasse_edges():
    """Build Hasse edges of the Boolean lattice (K-dim hypercube).

    Each edge connects two pipelines that differ by exactly one intervention.
    Returns undirected edges (src < dst to avoid duplicates).
    """
    edges = []
    for idx in range(N_PIPELINES):
        for k in range(K):
            neighbor = idx ^ (1 << k)
            if neighbor > idx:
                edges.append(InterventionEdge(
                    src=idx, dst=neighbor,
                    intervention=k, name=INTERVENTION_NAMES[k],
                ))
    return edges


def get_edge_tensors(edges):
    """Convert edges to tensors for efficient PGI loss computation."""
    src = torch.tensor([e.src for e in edges], dtype=torch.long)
    dst = torch.tensor([e.dst for e in edges], dtype=torch.long)
    iids = torch.tensor([e.intervention for e in edges], dtype=torch.long)
    return src, dst, iids


def compute_edge_weights(mode="uniform", kappa=None):
    """Compute per-intervention edge weights.

    Args:
        mode: "uniform" or "perturbation"
        kappa: (K,) perturbation magnitudes per intervention.

    Returns:
        weights: (K,) tensor, one weight per intervention type
    """
    if mode == "uniform":
        return torch.ones(K)

    if kappa is None:
        # Default perturbation magnitudes (heuristic, based on literature)
        kappa = torch.tensor([
            1.0,   # reference: moderate rank change
            1.5,   # hpf: significant low-freq content change
            1.2,   # lpf: high-freq content change
            0.8,   # baseline: moderate offset change
            2.0,   # asr: large signal modification
            1.0,   # autoreject: moderate trial modification
            0.7,   # badchannel: localized spatial change
        ])

    eps = 1e-6
    return 1.0 / (kappa + eps)


def pipeline_index_to_name(idx):
    """Convert pipeline index to human-readable name."""
    parts = []
    for i in range(K):
        bit = (idx >> i) & 1
        parts.append(f"{INTERVENTION_NAMES[i]}={INTERVENTION_OPTIONS[i][bit]}")
    return " | ".join(parts)


def get_intervention_edges(intervention_idx):
    """Get all edges that toggle a specific intervention.

    Useful for per-intervention sensitivity analysis.
    Returns list of (src, dst) pipeline index pairs.
    """
    pairs = []
    for idx in range(N_PIPELINES):
        neighbor = idx ^ (1 << intervention_idx)
        if neighbor > idx:
            pairs.append((idx, neighbor))
    return pairs
