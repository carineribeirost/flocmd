"""Periodic cluster analysis and aggregation kinetics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _wrap(pos: np.ndarray, box_lo: np.ndarray, box_len: np.ndarray) -> np.ndarray:
    x = np.mod(np.asarray(pos, dtype=float) - box_lo, box_len)
    return np.where(x >= box_len, 0.0, x)  # guard against x == L after rounding


def contact_pairs(pos, box_len, cutoff: float, box_lo=None) -> np.ndarray:
    """All pairs (i < j) closer than ``cutoff`` under periodic boundaries."""
    box_len = np.broadcast_to(np.asarray(box_len, dtype=float), (3,))
    lo = np.zeros(3) if box_lo is None else np.asarray(box_lo, dtype=float)
    if len(pos) < 2:
        return np.zeros((0, 2), dtype=int)
    tree = cKDTree(_wrap(pos, lo, box_len), boxsize=box_len)
    return tree.query_pairs(cutoff, output_type="ndarray")


def labels_from_pairs(n: int, pairs: np.ndarray) -> np.ndarray:
    """Connected-component labels (0..n_clusters-1) from an edge list."""
    if n == 0:
        return np.zeros(0, dtype=int)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def find_clusters(pos, box_len, cutoff: float, box_lo=None) -> np.ndarray:
    """Label particles by cluster; two particles belong together if linked by a chain of contacts."""
    return labels_from_pairs(len(pos), contact_pairs(pos, box_len, cutoff, box_lo))


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels) if len(labels) else np.zeros(0, dtype=int)


def size_moments(sizes: np.ndarray) -> dict[str, float]:
    """Number- and weight-averaged cluster sizes and the largest cluster."""
    sizes = np.asarray(sizes, dtype=float)
    if sizes.size == 0:
        return {"n_clusters": 0, "s_number": 0.0, "s_weight": 0.0, "s_max": 0}
    return {
        "n_clusters": int(len(sizes)),
        "s_number": float(sizes.mean()),
        "s_weight": float((sizes**2).sum() / sizes.sum()),
        "s_max": int(sizes.max()),
    }


def fit_smoluchowski(t, n_clusters, n0: int) -> tuple[float, float]:
    """Fit N(t) = N0 / (1 + t / tau) to the early-time cluster count.

    Returns (tau, tau_err). With number density n0 = N0 / V the aggregation
    rate constant is k = 2 / (n0 tau).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(n_clusters, dtype=float)

    def model(tt, tau):
        return n0 / (1.0 + tt / tau)

    guess = max(t[-1], 1.0) if len(t) else 1.0
    popt, pcov = curve_fit(model, t, y, p0=[guess], bounds=(1e-12, np.inf))
    return float(popt[0]), float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float("nan")


def radius_of_gyration_cluster(pos: np.ndarray, box_len) -> float:
    """Rg of a periodic cluster, unwrapped by walking from its first particle (valid if Rg << L)."""
    box_len = np.broadcast_to(np.asarray(box_len, dtype=float), (3,))
    ref = pos[0]
    d = pos - ref
    d -= box_len * np.round(d / box_len)
    x = ref + d
    return float(np.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()))
