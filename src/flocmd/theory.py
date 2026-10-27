"""Analytic references and reduced-unit <-> SI mapping.

Everything here is used either to choose physically meaningful simulation
parameters (Bjerrum/Debye lengths, DPD repulsions from chi) or to validate
simulation output against known results (DLVO pair potential, Groot–Warren
equation of state, Smoluchowski kinetics).
"""

from __future__ import annotations

import numpy as np

KB = 1.380649e-23          # J/K
E_CHARGE = 1.602176634e-19 # C
EPS0 = 8.8541878128e-12    # F/m
NA = 6.02214076e23         # 1/mol
R_GAS = KB * NA            # J/(mol K)


# --------------------------------------------------------------------------- #
# Electrostatics                                                              #
# --------------------------------------------------------------------------- #
def bjerrum_length_nm(T: float = 298.15, eps_r: float = 78.4) -> float:
    """Bjerrum length l_B = e^2 / (4 pi eps0 eps_r kT) in nm (0.715 nm for water at 25 C)."""
    return E_CHARGE**2 / (4 * np.pi * EPS0 * eps_r * KB * T) * 1e9


def debye_length_nm(ionic_strength_M: float, T: float = 298.15, eps_r: float = 78.4) -> float:
    """Debye length 1/kappa (nm) for ionic strength I (mol/L): kappa^2 = 8 pi l_B N_A I."""
    if ionic_strength_M <= 0:
        return np.inf
    lB = bjerrum_length_nm(T, eps_r) * 1e-9
    kappa2 = 8 * np.pi * lB * NA * ionic_strength_M * 1e3
    return 1e9 / np.sqrt(kappa2)


def ionic_strength_from_debye_M(debye_nm: float, T: float = 298.15, eps_r: float = 78.4) -> float:
    lB = bjerrum_length_nm(T, eps_r) * 1e-9
    return 1.0 / ((debye_nm * 1e-9) ** 2 * 8 * np.pi * lB * NA * 1e3)


def dlvo_charge_factor(kappa: float, radius: float) -> float:
    """Finite-size factor e^{ka}/(1+ka) of the linearised Poisson–Boltzmann (DLVO) Yukawa potential."""
    return np.exp(kappa * radius) / (1.0 + kappa * radius)


def yukawa_dlvo(r, z1, z2, a1, a2, debye, lB=1.0):
    """Screened-Coulomb interaction (kT) between two charged spheres of radii a1, a2.

    U(r) = l_B z1 z2 [e^{k a1}/(1+k a1)] [e^{k a2}/(1+k a2)] e^{-k r} / r
    """
    kappa = 1.0 / debye
    r = np.asarray(r, dtype=float)
    pref = lB * z1 * z2 * dlvo_charge_factor(kappa, a1) * dlvo_charge_factor(kappa, a2)
    return pref * np.exp(-kappa * r) / r


def lj_expand(r, eps=1.0, sigma=1.0, delta=0.0, rc=2.0 ** (1 / 6), shift=True):
    """LAMMPS ``lj/expand``: LJ in (r - delta), cut at rc + delta, optionally energy-shifted."""
    r = np.asarray(r, dtype=float)
    s = r - delta
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 4 * eps * ((sigma / s) ** 12 - (sigma / s) ** 6)
    if shift:
        u = u - 4 * eps * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
    u = np.where(s < rc, u, 0.0)
    return np.where(s > 0, u, np.inf)


def colloid_pair_potential(r, dc=6.0, zc=20.0, debye=2.0, lB=1.0):
    """Bare colloid–colloid potential of Module 1 (WCA core + DLVO Yukawa), in kT.

    This is the exact PMF of the two-colloid system *without* polymer and is used
    to validate the umbrella-sampling/WHAM pipeline of Module 2.
    """
    a = 0.5 * dc
    return lj_expand(r, 1.0, 1.0, dc - 1.0) + yukawa_dlvo(r, zc, zc, a, a, debye, lB)


# --------------------------------------------------------------------------- #
# Aggregation kinetics                                                        #
# --------------------------------------------------------------------------- #
def smoluchowski_fast_rate(D: float, radius: float) -> float:
    """Diffusion-limited rate constant k_11 = 4 pi (D1+D2)(a1+a2) = 16 pi D a for identical spheres."""
    return 16.0 * np.pi * D * radius


def smoluchowski_n_clusters(t, n0: float, k: float, volume: float):
    """Total cluster count for constant-kernel Smoluchowski kinetics: N(t) = N0 / (1 + k n0 t / 2)."""
    return n0 / (1.0 + 0.5 * k * (n0 / volume) * np.asarray(t, dtype=float))


# --------------------------------------------------------------------------- #
# DPD                                                                         #
# --------------------------------------------------------------------------- #
def groot_warren_pressure(rho, a, alpha: float = 0.101):
    """Groot–Warren equation of state p = rho kT + alpha a rho^2 (kT = rc = 1, valid for rho > 2)."""
    rho = np.asarray(rho, dtype=float)
    return rho + alpha * np.asarray(a, dtype=float) * rho**2


def dpd_repulsion_from_chi(chi: float, a_ii: float = 25.0) -> float:
    """a_ij = a_ii + 3.27 chi_ij for rho = 3 (Groot & Madden 1998; Groot & Rabone 2001)."""
    return a_ii + 3.27 * chi


def chi_from_solubility(delta_i: float, delta_j: float, v_bead_cm3_mol: float, T: float = 298.15) -> float:
    """Hildebrand estimate chi = V_bead (delta_i - delta_j)^2 / RT, delta in (J/cm^3)^1/2."""
    return v_bead_cm3_mol * (delta_i - delta_j) ** 2 / (R_GAS * T)


def dpd_cutoff_nm(rho: float = 3.0, n_water_per_bead: int = 3) -> float:
    """DPD length unit r_c = 3.107 (rho N_m)^{1/3} Angstrom (Groot & Rabone 2001)."""
    return 0.3107 * (rho * n_water_per_bead) ** (1.0 / 3.0)
