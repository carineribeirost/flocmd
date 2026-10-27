"""Minimal writer for LAMMPS data files (``atom_style full``).

Builders produce *unwrapped* coordinates; the writer wraps them into the
primary box and stores the image flags, so bonded molecules that cross a
periodic boundary are reconstructed correctly by LAMMPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LammpsSystem:
    box_lo: np.ndarray
    box_hi: np.ndarray
    positions: np.ndarray            # (N, 3), may be unwrapped
    types: np.ndarray                # (N,), 1-based atom types
    mol: np.ndarray                  # (N,), molecule IDs (0 = no molecule)
    charges: np.ndarray              # (N,)
    masses: dict[int, float]
    n_atom_types: int
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int))  # (M, 3): type, i, j (0-based)
    n_bond_types: int = 0
    type_names: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.box_lo = np.asarray(self.box_lo, dtype=float)
        self.box_hi = np.asarray(self.box_hi, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        self.types = np.asarray(self.types, dtype=int)
        self.mol = np.asarray(self.mol, dtype=int)
        self.charges = np.asarray(self.charges, dtype=float)
        self.bonds = np.asarray(self.bonds, dtype=int).reshape(-1, 3)
        n = len(self.positions)
        for name in ("types", "mol", "charges"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")
        if n and (self.types.min() < 1 or self.types.max() > self.n_atom_types):
            raise ValueError("atom types must lie in 1..n_atom_types")
        if len(self.bonds) and (self.bonds[:, 1:].min() < 0 or self.bonds[:, 1:].max() >= n):
            raise ValueError("bond indices out of range")

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def box_lengths(self) -> np.ndarray:
        return self.box_hi - self.box_lo

    @property
    def net_charge(self) -> float:
        return float(self.charges.sum())

    def wrapped(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (wrapped positions, integer image flags)."""
        L = self.box_lengths
        image = np.floor((self.positions - self.box_lo) / L).astype(int)
        return self.positions - image * L, image

    def write(self, path: str | Path, title: str = "flocmd generated data file") -> Path:
        path = Path(path)
        x, image = self.wrapped()
        lines = [f"# {title}", ""]
        lines.append(f"{self.n_atoms} atoms")
        lines.append(f"{len(self.bonds)} bonds")
        lines.append(f"{self.n_atom_types} atom types")
        lines.append(f"{max(self.n_bond_types, 1 if len(self.bonds) else 0)} bond types")
        lines.append("")
        for k, axis in enumerate("xyz"):
            lines.append(f"{self.box_lo[k]:.6f} {self.box_hi[k]:.6f} {axis}lo {axis}hi")
        lines += ["", "Masses", ""]
        for t in range(1, self.n_atom_types + 1):
            label = f"  # {self.type_names[t]}" if t in self.type_names else ""
            lines.append(f"{t} {self.masses[t]:.6f}{label}")
        lines += ["", "Atoms # full", ""]
        lines += [
            f"{i + 1} {self.mol[i]} {self.types[i]} {self.charges[i]:.6f} "
            f"{x[i, 0]:.6f} {x[i, 1]:.6f} {x[i, 2]:.6f} {image[i, 0]} {image[i, 1]} {image[i, 2]}"
            for i in range(self.n_atoms)
        ]
        if len(self.bonds):
            lines += ["", "Bonds", ""]
            lines += [f"{k + 1} {b[0]} {b[1] + 1} {b[2] + 1}" for k, b in enumerate(self.bonds)]
        path.write_text("\n".join(lines) + "\n")
        return path


def min_image(d: np.ndarray, L: np.ndarray | float) -> np.ndarray:
    """Apply the minimum-image convention to displacement vectors."""
    return d - L * np.round(d / L)
