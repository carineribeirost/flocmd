"""Builder for Modules 1–2: charged nanocolloids + cationic polyelectrolyte flocculant.

Coarse-grained, implicit-solvent model (LJ units, sigma ~ 0.71 nm so that the
Bjerrum length of water is ~1 sigma):

* type 1  colloid, diameter ``dc`` sigma, bare charge ``-Z`` e (e.g. silica, clay, oil-wet fines)
* type 2  neutral monomer (acrylamide-like)
* type 3  cationic monomer, +1 e (e.g. AETAC/DMC units of a cationic polyacrylamide, CPAM)

Chains are Kremer–Grest bead–spring polymers. The polymer *dose* is expressed
as the charge ratio  Phi = (total polymer charge) / |total colloid charge|,
the natural variable for charge-neutralisation (Phi ~ 1) and overdosing (Phi > 1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..lammps_data import LammpsSystem, min_image

COLLOID, NEUTRAL, CATIONIC = 1, 2, 3


@dataclass
class ColloidPEParams:
    n_colloids: int = 27
    colloid_diameter: float = 6.0       # sigma
    colloid_charge: float = 20.0        # |Z|, bare charge per colloid (e)
    volume_fraction: float = 0.05       # colloid volume fraction (sets the box if box_length is None)
    box_length: float | None = None     # sigma; overrides volume_fraction
    chain_length: int = 60              # beads per chain
    charge_fraction: float = 0.5        # fraction of cationic beads (CPAM "charge density")
    charge_pattern: str = "regular"     # "regular" | "random"
    dose: float = 1.0                   # charge ratio Phi; ignored if n_chains is given
    n_chains: int | None = None
    colloid_separation: float | None = None  # PMF mode: exactly two colloids at this distance along x
    min_gap: float = 3.0                # minimum surface-surface gap at t = 0 (> floc contact gap)
    bond_length: float = 0.97
    seed: int = 1


def charge_sequence(n: int, fraction: float, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of charged beads along a chain of length n."""
    n_q = int(round(fraction * n))
    if fraction > 0 and n_q == 0:
        n_q = 1
    mask = np.zeros(n, dtype=bool)
    if n_q == 0:
        return mask
    if pattern == "regular":
        idx = np.floor((np.arange(n_q) + 0.5) * n / n_q).astype(int)
    elif pattern == "random":
        idx = rng.choice(n, size=n_q, replace=False)
    else:
        raise ValueError(f"unknown charge_pattern {pattern!r}")
    mask[idx] = True
    return mask


def n_chains_for_dose(p: ColloidPEParams) -> int:
    if p.n_chains is not None:
        return int(p.n_chains)
    n_q = int(charge_sequence(p.chain_length, p.charge_fraction, "regular", np.random.default_rng(0)).sum())
    if n_q == 0:
        raise ValueError("a neutral polymer needs an explicit n_chains (dose is a charge ratio)")
    return max(0, int(round(p.dose * p.n_colloids * p.colloid_charge / n_q)))


def _box_length(p: ColloidPEParams) -> float:
    if p.box_length is not None:
        return float(p.box_length)
    if p.n_colloids == 0:
        raise ValueError("box_length is required when there are no colloids")
    v_c = np.pi / 6.0 * p.colloid_diameter**3
    return float((p.n_colloids * v_c / p.volume_fraction) ** (1.0 / 3.0))


def _place_colloids(p: ColloidPEParams, L: float, rng: np.random.Generator) -> np.ndarray:
    if p.colloid_separation is not None:
        if p.n_colloids != 2:
            raise ValueError("colloid_separation requires n_colloids = 2")
        c = np.full(3, 0.5 * L)
        h = 0.5 * p.colloid_separation
        return np.array([c - [h, 0, 0], c + [h, 0, 0]])
    dmin = p.colloid_diameter + p.min_gap
    centres: list[np.ndarray] = []
    for _ in range(200_000):
        if len(centres) == p.n_colloids:
            break
        trial = rng.uniform(0, L, 3)
        if centres:
            d = np.linalg.norm(min_image(np.asarray(centres) - trial, L), axis=1)
            if d.min() < dmin:
                continue
        centres.append(trial)
    if len(centres) < p.n_colloids:
        raise RuntimeError("could not place colloids; lower volume_fraction or min_gap")
    return np.asarray(centres).reshape(-1, 3)


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _grow_chain(n: int, L: float, colloids: np.ndarray, r_excl: float, b: float,
                rng: np.random.Generator) -> np.ndarray:
    """Self-avoiding-ish random walk that stays out of the colloid cores."""

    def outside(x: np.ndarray) -> bool:
        if len(colloids) == 0:
            return True
        return np.linalg.norm(min_image(colloids - x, L), axis=1).min() >= r_excl

    for _restart in range(1000):
        chain = np.empty((n, 3))
        x = rng.uniform(0, L, 3)
        if not outside(x):
            continue
        chain[0] = x
        ok = True
        for i in range(1, n):
            for _try in range(200):
                trial = chain[i - 1] + b * _random_unit(rng)
                if not outside(trial):
                    continue
                if i >= 2 and np.linalg.norm(trial - chain[i - 2]) < 1.0:
                    continue
                chain[i] = trial
                break
            else:
                ok = False
                break
        if ok:
            return chain
    raise RuntimeError("failed to grow a chain; the box is too crowded")


def build(p: ColloidPEParams) -> tuple[LammpsSystem, dict]:
    rng = np.random.default_rng(p.seed)
    L = _box_length(p)
    centres = _place_colloids(p, L, rng)
    n_chains = n_chains_for_dose(p)
    r_excl = 0.5 * p.colloid_diameter + 0.5 + 0.1

    pos = [centres]
    types = [np.full(len(centres), COLLOID)]
    charges = [np.full(len(centres), -float(p.colloid_charge))]
    mol = [np.arange(1, len(centres) + 1)]
    bonds = []
    sequences = []
    offset = len(centres)
    for c in range(n_chains):
        seq = charge_sequence(p.chain_length, p.charge_fraction, p.charge_pattern, rng)
        sequences.append(seq.astype(int).tolist())
        xyz = _grow_chain(p.chain_length, L, centres, r_excl, p.bond_length, rng)
        pos.append(xyz)
        types.append(np.where(seq, CATIONIC, NEUTRAL))
        charges.append(seq.astype(float))
        mol.append(np.full(p.chain_length, len(centres) + c + 1))
        i = offset + np.arange(p.chain_length - 1)
        bonds.append(np.column_stack([np.ones_like(i), i, i + 1]))
        offset += p.chain_length

    system = LammpsSystem(
        box_lo=np.zeros(3), box_hi=np.full(3, L),
        positions=np.vstack(pos), types=np.concatenate(types), mol=np.concatenate(mol),
        charges=np.concatenate(charges),
        masses={COLLOID: p.colloid_diameter**3, NEUTRAL: 1.0, CATIONIC: 1.0},
        n_atom_types=3,
        bonds=np.vstack(bonds) if bonds else np.zeros((0, 3), dtype=int),
        n_bond_types=1,
        type_names={COLLOID: "colloid", NEUTRAL: "neutral monomer", CATIONIC: "cationic monomer"},
    )
    q_poly = float(sum(np.sum(s) for s in sequences))
    q_coll = p.n_colloids * p.colloid_charge
    meta = {
        "builder": "colloid_pe",
        "params": asdict(p),
        "box_length": L,
        "n_colloids": int(len(centres)),
        "n_chains": n_chains,
        "n_atoms": system.n_atoms,
        "dose_actual": q_poly / q_coll if q_coll > 0 else None,
        "volume_fraction_actual": len(centres) * np.pi / 6 * p.colloid_diameter**3 / L**3,
        "net_bare_charge": system.net_charge,
        "sequences": sequences,
    }
    return system, meta


def build_single_chain(n: int, seed: int = 1) -> tuple[LammpsSystem, dict]:
    """One neutral Kremer–Grest chain in a box large enough to avoid self-images (validation)."""
    L = 2.0 * n**0.6 + 20.0
    p = ColloidPEParams(n_colloids=0, box_length=L, chain_length=n, charge_fraction=0.0,
                        n_chains=1, seed=seed)
    return build(p)
