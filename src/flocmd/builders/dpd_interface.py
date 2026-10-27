"""Builder for Module 3: DPD water | oil | water slab with flocculant chains.

Bead types (Groot–Warren DPD, rho = 3, N_m = 3 water molecules per bead, r_c ~ 0.65 nm):

* 1  W  water
* 2  O  oil (molecules are linear ``oil_length``-mers; 3 beads ~ n-decane/dodecane)
* 3  P  hydrophilic backbone bead (acrylamide-like)
* 4  H  hydrophobic sticker (alkyl side group of hydrophobically modified PAM, HM-PAM)

The oil slab is centred at L_z/2 with its normal along z, giving two
planar interfaces in a periodic box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..lammps_data import LammpsSystem

WATER, OIL, BACKBONE, STICKER = 1, 2, 3, 4
ARCHITECTURES = ("homopolymer", "random", "blocky", "telechelic", "diblock")


@dataclass
class DPDInterfaceParams:
    box: tuple[float, float, float] = (20.0, 20.0, 40.0)
    density: float = 3.0
    oil_thickness_fraction: float = 0.4
    oil_length: int = 3
    n_chains: int = 20
    chain_length: int = 30
    architecture: str = "random"
    hydrophobic_fraction: float = 0.2
    block_length: int = 3
    polymer_start: str = "interface"   # "interface" (near both interfaces) | "water" (bulk water)
    bond_length: float = 0.8
    seed: int = 1


def sticker_sequence(n: int, architecture: str, fraction: float, block: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of hydrophobic (H) beads along a chain of length n."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    mask = np.zeros(n, dtype=bool)
    n_h = 0 if architecture == "homopolymer" else int(round(fraction * n))
    if n_h == 0:
        return mask
    if architecture == "random":
        mask[rng.choice(n, size=n_h, replace=False)] = True
    elif architecture == "blocky":
        n_blocks = max(1, int(round(n_h / block)))
        for k in range(n_blocks):
            start = int(np.floor((k + 0.5) * n / n_blocks - 0.5 * block))
            mask[max(0, start):min(n, start + block)] = True
    elif architecture == "telechelic":
        mask[: n_h // 2 + n_h % 2] = True
        mask[n - n_h // 2:] = True
    elif architecture == "diblock":
        mask[:n_h] = True
    return mask


class _Slab:
    """Geometry helper for a periodic water | oil | water slab along z."""

    def __init__(self, box: np.ndarray, oil_fraction: float) -> None:
        self.box = box
        h = oil_fraction * box[2]
        self.z1, self.z2 = 0.5 * (box[2] - h), 0.5 * (box[2] + h)

    def in_oil(self, z: float) -> bool:
        zw = z % self.box[2]
        return self.z1 <= zw <= self.z2

    def random_water_point(self, rng: np.random.Generator, depth: float | None = None) -> np.ndarray:
        """Uniform point in water; if depth is given, within that distance of an interface."""
        lx, ly, lz = self.box
        if depth is None:
            z = (self.z2 + rng.uniform(0, lz - (self.z2 - self.z1))) % lz
        else:
            z = self.z1 - rng.uniform(0, depth) if rng.random() < 0.5 else self.z2 + rng.uniform(0, depth)
        return np.array([rng.uniform(0, lx), rng.uniform(0, ly), z])

    def random_oil_point(self, rng: np.random.Generator) -> np.ndarray:
        lx, ly, _ = self.box
        return np.array([rng.uniform(0, lx), rng.uniform(0, ly), rng.uniform(self.z1, self.z2)])


def _walk(start: np.ndarray, n: int, b: float, accept, rng: np.random.Generator) -> np.ndarray:
    xyz = np.empty((n, 3))
    xyz[0] = start
    for i in range(1, n):
        for _ in range(500):
            v = rng.normal(size=3)
            trial = xyz[i - 1] + b * v / np.linalg.norm(v)
            if accept(trial):
                break
        xyz[i] = trial  # after 500 failures keep the last trial; DPD forces are soft
    return xyz


def build(p: DPDInterfaceParams) -> tuple[LammpsSystem, dict]:
    rng = np.random.default_rng(p.seed)
    box = np.asarray(p.box, dtype=float)
    slab = _Slab(box, p.oil_thickness_fraction)
    volume = float(np.prod(box))
    n_total = int(round(p.density * volume))
    n_oil_mol = int(round(p.density * box[0] * box[1] * (slab.z2 - slab.z1) / p.oil_length))
    n_poly = p.n_chains * p.chain_length
    n_water = n_total - n_oil_mol * p.oil_length - n_poly
    if n_water <= 0:
        raise ValueError("not enough volume left for water")

    pos, types, mol, bonds = [], [], [], []
    offset, mol_id = 0, 0

    for _ in range(n_oil_mol):
        mol_id += 1
        xyz = _walk(slab.random_oil_point(rng), p.oil_length, p.bond_length,
                    lambda x: slab.in_oil(x[2]), rng)
        pos.append(xyz)
        types.append(np.full(p.oil_length, OIL))
        mol.append(np.full(p.oil_length, mol_id))
        i = offset + np.arange(p.oil_length - 1)
        bonds.append(np.column_stack([np.full_like(i, 1), i, i + 1]))
        offset += p.oil_length

    first_polymer_mol = mol_id + 1
    sequences = []
    depth = 3.0 if p.polymer_start == "interface" else None
    for _ in range(p.n_chains):
        mol_id += 1
        seq = sticker_sequence(p.chain_length, p.architecture, p.hydrophobic_fraction, p.block_length, rng)
        sequences.append(seq.astype(int).tolist())
        xyz = _walk(slab.random_water_point(rng, depth), p.chain_length, p.bond_length,
                    lambda x: not slab.in_oil(x[2]), rng)
        pos.append(xyz)
        types.append(np.where(seq, STICKER, BACKBONE))
        mol.append(np.full(p.chain_length, mol_id))
        i = offset + np.arange(p.chain_length - 1)
        bonds.append(np.column_stack([np.full_like(i, 2), i, i + 1]))
        offset += p.chain_length

    water = np.array([slab.random_water_point(rng) for _ in range(n_water)])
    pos.append(water)
    types.append(np.full(n_water, WATER))
    mol.append(np.zeros(n_water, dtype=int))

    positions = np.vstack(pos)
    system = LammpsSystem(
        box_lo=np.zeros(3), box_hi=box, positions=positions,
        types=np.concatenate(types), mol=np.concatenate(mol), charges=np.zeros(len(positions)),
        masses={WATER: 1.0, OIL: 1.0, BACKBONE: 1.0, STICKER: 1.0}, n_atom_types=4,
        bonds=np.vstack(bonds) if bonds else np.zeros((0, 3), dtype=int), n_bond_types=2,
        type_names={WATER: "W water", OIL: "O oil", BACKBONE: "P backbone", STICKER: "H sticker"},
    )
    meta = {
        "builder": "dpd_interface",
        "params": asdict(p),
        "box": box.tolist(),
        "n_atoms": system.n_atoms,
        "n_water": n_water,
        "n_oil_molecules": n_oil_mol,
        "first_polymer_mol": first_polymer_mol,
        "n_chains": p.n_chains,
        "slab_z": [slab.z1, slab.z2],
        "sequences": sequences,
        "hydrophobic_fraction_actual": float(np.mean([np.mean(s) for s in sequences])) if sequences else 0.0,
    }
    return system, meta
