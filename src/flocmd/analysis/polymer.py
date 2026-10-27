"""Chain conformation: radius of gyration (tensor), anisotropy at interfaces, Flory exponent."""

from __future__ import annotations

import numpy as np


def gyration_tensors(pos_unwrapped: np.ndarray, mol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-molecule centre of mass (n, 3) and gyration tensor (n, 3, 3), equal masses."""
    ids = np.unique(mol)
    com = np.empty((len(ids), 3))
    S = np.empty((len(ids), 3, 3))
    for k, m in enumerate(ids):
        x = pos_unwrapped[mol == m]
        com[k] = x.mean(axis=0)
        d = x - com[k]
        S[k] = d.T @ d / len(x)
    return com, S


def rg_components(S: np.ndarray) -> dict[str, np.ndarray]:
    """Rg^2, in-plane Rg_par^2 = Sxx + Syy, normal Rg_perp^2 = Szz, and anisotropy.

    anisotropy = sqrt(Rg_par^2 / (2 Rg_perp^2)) equals 1 for an isotropic coil and
    grows when a chain flattens on a plane normal to z (adsorption at an interface).
    """
    rg2 = np.trace(S, axis1=1, axis2=2)
    par = S[:, 0, 0] + S[:, 1, 1]
    perp = S[:, 2, 2]
    return {"rg2": rg2, "rg2_par": par, "rg2_perp": perp,
            "anisotropy": np.sqrt(par / (2.0 * np.maximum(perp, 1e-12)))}


def flory_exponent(n, rg, rg_err=None) -> tuple[float, float, float]:
    """Fit Rg = b N^nu in log–log space; returns (nu, nu_err, prefactor b)."""
    n = np.asarray(n, dtype=float)
    rg = np.asarray(rg, dtype=float)
    w = None
    if rg_err is not None:
        w = 1.0 / np.maximum(np.asarray(rg_err, dtype=float) / rg, 1e-12)  # sigma of log(rg)
    coef, cov = np.polyfit(np.log(n), np.log(rg), 1, w=w, cov="unscaled" if w is not None else True)
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(np.exp(coef[1]))
