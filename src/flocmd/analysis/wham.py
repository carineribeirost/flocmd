"""One-dimensional WHAM (Kumar et al. 1992) with block-bootstrap errors.

Each window w samples the biased distribution of the reaction coordinate r
under U_w(r) = K_w/2 (r - r0_w)^2. WHAM finds the unbiased density P(r) and
window free energies f_w self-consistently:

    P(b)        = sum_w n_w(b) / sum_w N_w exp[-(U_w(b) - f_w)/kT]
    exp(-f_w/kT) = sum_b P(b) exp[-U_w(b)/kT]

For a centre-of-mass distance in 3D the free energy contains the entropic
2 kT ln r term from the r^2 dr volume element; the pair PMF is
w(r) = -kT ln P(r) + 2 kT ln r, comparable to a pair potential.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .stats import statistical_inefficiency


@dataclass
class WhamResult:
    centers: np.ndarray
    free_energy: np.ndarray      # -kT ln P(r), shifted to 0 at the reference
    pmf: np.ndarray              # with the Jacobian correction (pair PMF)
    f_windows: np.ndarray
    iterations: int
    pmf_err: np.ndarray | None = None
    replicates: np.ndarray | None = None   # bootstrap PMFs (n_boot, bins), NaN where unsampled


def wham(samples: list[np.ndarray], r0: np.ndarray, k: np.ndarray, kT: float = 1.0,
         bins: int | np.ndarray = 100, tol: float = 1e-7, max_iter: int = 100_000,
         correlated: bool = True, jacobian: bool = True, ref: str = "far") -> WhamResult:
    r0 = np.asarray(r0, dtype=float)
    k = np.broadcast_to(np.asarray(k, dtype=float), r0.shape)
    all_r = np.concatenate(samples)
    edges = np.histogram_bin_edges(all_r, bins=bins) if np.isscalar(bins) else np.asarray(bins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    g = np.array([statistical_inefficiency(s) if correlated else 1.0 for s in samples])
    hist = np.array([np.histogram(s, bins=edges)[0] for s in samples], dtype=float) / g[:, None]
    n_eff = hist.sum(axis=1)
    bias = 0.5 * k[:, None] * (centers[None, :] - r0[:, None]) ** 2 / kT     # (W, B)
    numer = hist.sum(axis=0)

    f = np.zeros(len(samples))
    for it in range(1, max_iter + 1):
        denom = (n_eff[:, None] * np.exp(-bias + f[:, None])).sum(axis=0)
        p = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        f_new = -np.log((p[None, :] * np.exp(-bias)).sum(axis=1))
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    with np.errstate(divide="ignore"):
        fe = -kT * np.log(p)
        pmf = fe + (2.0 * kT * np.log(centers) if jacobian else 0.0)
    fe, pmf = _shift(fe, ref), _shift(pmf, ref)
    return WhamResult(centers, fe, pmf, f * kT, it)


def _shift(y: np.ndarray, ref: str) -> np.ndarray:
    finite = np.isfinite(y)
    if not finite.any():
        return y
    if ref == "far":
        tail = np.flatnonzero(finite)[-max(1, finite.sum() // 20):]
        return y - y[tail].mean()
    return y - y[finite].min()


def bootstrap_wham(samples, r0, k, n_boot: int = 50, block: int | None = None, seed: int = 0,
                   **kwargs) -> WhamResult:
    """WHAM plus a moving-block bootstrap of every window (preserves autocorrelation)."""
    rng = np.random.default_rng(seed)
    base = wham(samples, r0, k, **kwargs)
    kwargs = {**kwargs, "bins": np.concatenate([base.centers - 0.5 * np.diff(base.centers)[0],
                                                [base.centers[-1] + 0.5 * np.diff(base.centers)[0]]])}
    reps = []
    for _ in range(n_boot):
        resampled = []
        for s in samples:
            b = block or max(1, int(np.ceil(statistical_inefficiency(s))))
            n_blocks = int(np.ceil(len(s) / b))
            starts = rng.integers(0, max(1, len(s) - b + 1), size=n_blocks)
            resampled.append(np.concatenate([s[a:a + b] for a in starts])[: len(s)])
        reps.append(wham(resampled, r0, k, **kwargs).pmf)
    reps = np.array(reps)
    base.replicates = np.where(np.isfinite(reps), reps, np.nan)
    base.pmf_err = spread(base.replicates)
    return base


def spread(replicates: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Bootstrap standard deviation per bin, after shifting every replica to zero mean on ``ref`` bins."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # bins never sampled in some replicas
        if ref is not None:
            replicates = replicates - np.nanmean(replicates[:, ref], axis=1, keepdims=True)
        return np.nanstd(replicates, axis=0)
