"""Planar-interface analysis for the DPD water | oil | water slab (Module 3)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def slab_profile(z, rho_in, rho_out, z1, z2, w):
    """Density of a slab between interfaces z1 < z2 with tanh width w."""
    return rho_out + 0.5 * (rho_in - rho_out) * (np.tanh((z - z1) / w) - np.tanh((z - z2) / w))


def fit_slab(z, rho) -> dict[str, float]:
    """Fit a two-interface tanh profile; returns rho_in, rho_out, z1, z2 (Gibbs surfaces) and width w."""
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    half = 0.5 * (rho.max() + rho.min())
    inside = z[rho > half]
    p0 = [rho.max(), rho.min(), inside.min(), inside.max(), 1.0]
    popt, pcov = curve_fit(slab_profile, z, rho, p0=p0, maxfev=20000)
    err = np.sqrt(np.diag(pcov))
    keys = ["rho_in", "rho_out", "z1", "z2", "width"]
    out = {k: float(v) for k, v in zip(keys, popt)}
    out.update({f"{k}_err": float(e) for k, e in zip(keys, err)})
    out["width"] = abs(out["width"])
    return out


def interfacial_tension(pxx, pyy, pzz, lz: float, n_interfaces: int = 2):
    """Mechanical (Kirkwood–Buff) route: gamma = Lz/n [Pzz - (Pxx + Pyy)/2]."""
    return lz / n_interfaces * (np.asarray(pzz) - 0.5 * (np.asarray(pxx) + np.asarray(pyy)))


def mean_profile(blocks, discard: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average ``fix ave/chunk density/number`` blocks: returns z, mean density, SEM across blocks."""
    rows = [b for _, b in blocks]
    rows = rows[int(discard * len(rows)):]
    z = rows[0][:, 1]
    dens = np.array([r[:, -1] for r in rows])
    sem = dens.std(axis=0, ddof=1) / np.sqrt(len(dens)) if len(dens) > 1 else np.zeros_like(z)
    return z, dens.mean(axis=0), sem


def distance_to_interfaces(z, z1: float, z2: float, lz: float) -> np.ndarray:
    """Signed distance to the nearest interface: > 0 in water, < 0 inside the oil slab (periodic z)."""
    zw = np.mod(np.asarray(z, dtype=float), lz)
    d1 = zw - z1
    d2 = zw - z2
    d1 -= lz * np.round(d1 / lz)
    d2 -= lz * np.round(d2 / lz)
    nearest = np.where(np.abs(d1) < np.abs(d2), -d1, d2)   # water side positive for both interfaces
    return nearest


def surface_excess(z, rho, z1: float, z2: float, lz: float, area: float, bulk_rho: float = 0.0,
                   window: float = 2.0) -> float:
    """Beads per unit area within +-window of the two interfaces, minus the bulk contribution."""
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    dz = z[1] - z[0]
    near = np.abs(distance_to_interfaces(z, z1, z2, lz)) <= window
    return float(((rho[near] - bulk_rho) * dz).sum() * area / (2.0 * area))
