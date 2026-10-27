"""Error analysis for correlated time series.

Molecular-dynamics samples are correlated, so naive standard errors are too
small. We use Flyvbjerg–Petersen block averaging and the integrated
autocorrelation time with Sokal's automatic window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockResult:
    mean: float
    sem: float
    block_sizes: np.ndarray
    sems: np.ndarray
    sem_errors: np.ndarray
    plateau_level: int


def block_average(x, min_blocks: int = 8) -> BlockResult:
    """Flyvbjerg–Petersen blocking.

    At each level the series is coarse-grained by averaging neighbouring pairs.
    The SEM estimate rises with block size and plateaus once blocks are longer
    than the correlation time; the plateau is taken as the first level whose
    estimate agrees, within its own error bar, with all larger-block estimates.
    """
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    sizes, sems, errs = [], [], []
    y, size = x.copy(), 1
    while len(y) >= min_blocks:
        n = len(y)
        var = y.var(ddof=1) if n > 1 else 0.0
        sem = np.sqrt(var / n)
        sizes.append(size)
        sems.append(sem)
        errs.append(sem / np.sqrt(2.0 * (n - 1)))
        if n % 2:
            y = y[:-1]
        y = 0.5 * (y[0::2] + y[1::2])
        size *= 2
    sems_a, errs_a = np.array(sems), np.array(errs)
    level = len(sems_a) - 1
    for i in range(len(sems_a)):
        later = sems_a[i:]
        if np.all(np.abs(later - sems_a[i]) <= errs_a[i] + errs_a[i:]):
            level = i
            break
    sem_est = float(sems_a[level]) if len(sems_a) else float("nan")
    return BlockResult(mean, sem_est, np.array(sizes), sems_a, errs_a, level)


def autocorrelation(x) -> np.ndarray:
    """Normalised autocorrelation function via FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0] if acf[0] > 0 else acf


def integrated_autocorr_time(x, c: float = 5.0) -> float:
    """tau_int = 1/2 + sum_t rho(t), truncated at the smallest window M >= c tau_int (Sokal)."""
    rho = autocorrelation(x)
    tau = 0.5 + np.cumsum(rho[1:])
    for m in range(1, len(tau)):
        if m >= c * tau[m - 1]:
            return float(tau[m - 1])
    return float(tau[-1])


def statistical_inefficiency(x) -> float:
    """g = 2 tau_int (>= 1): number of correlated samples per independent sample."""
    return max(1.0, 2.0 * integrated_autocorr_time(x))


def mean_sem(x, discard: float = 0.0) -> tuple[float, float]:
    """Mean and block-averaged SEM after discarding the first fraction of samples."""
    x = np.asarray(x, dtype=float)
    x = x[int(discard * len(x)):]
    r = block_average(x)
    return r.mean, r.sem
