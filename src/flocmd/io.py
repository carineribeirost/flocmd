"""Readers for LAMMPS output: thermo logs, fix ave/time, fix ave/chunk and custom dumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def read_log(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Parse every thermo block of a LAMMPS log into ``{column: array}`` dicts."""
    blocks, header, rows = [], None, []
    for line in Path(path).read_text(errors="replace").splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "Step":
            header, rows = tokens, []
            continue
        if header is None:
            continue
        if line.startswith("Loop time") or not tokens:
            if rows:
                arr = np.array(rows, dtype=float)
                blocks.append({h: arr[:, i] for i, h in enumerate(header)})
            header, rows = None, []
            continue
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            continue  # warnings interleaved with thermo output
        if len(values) == len(header):
            rows.append(values)
    return blocks


def read_ave_time(path: str | Path) -> dict[str, np.ndarray]:
    """Read a scalar-mode ``fix ave/time`` file into ``{column: array}``."""
    names: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                names = line[1:].split()
            else:
                break
    data = np.loadtxt(path, comments="#", ndmin=2)
    if not names or len(names) != data.shape[1]:
        names = [f"c{i}" for i in range(data.shape[1])]
    return {n: data[:, i] for i, n in enumerate(names)}


def read_ave_blocks(path: str | Path) -> list[tuple[int, np.ndarray]]:
    """Read vector-mode ``fix ave/time`` or ``fix ave/chunk`` files.

    Both formats consist of blocks headed by ``timestep n_rows [extra]``
    followed by ``n_rows`` numeric lines. Returns ``[(timestep, rows), ...]``.
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    blocks, i = [], 0
    while i < len(lines):
        head = lines[i].split()
        step, n = int(float(head[0])), int(float(head[1]))
        rows = np.array([lines[i + 1 + k].split() for k in range(n)], dtype=float)
        blocks.append((step, rows))
        i += n + 1
    return blocks


@dataclass
class Frame:
    timestep: int
    box_lo: np.ndarray
    box_hi: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def box_lengths(self) -> np.ndarray:
        return self.box_hi - self.box_lo

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def positions(self, kind: str = "x") -> np.ndarray:
        """Return (N, 3) coordinates: 'x' wrapped, 'xu' unwrapped (from xu or x + image*L)."""
        c = self.columns
        if kind == "xu":
            if "xu" in c:
                return np.column_stack([c["xu"], c["yu"], c["zu"]])
            x = np.column_stack([c["x"], c["y"], c["z"]])
            if "ix" in c:
                x = x + np.column_stack([c["ix"], c["iy"], c["iz"]]) * self.box_lengths
            return x
        return np.column_stack([c["x"], c["y"], c["z"]])


def iter_dump(path: str | Path, every: int = 1) -> Iterator[Frame]:
    """Iterate over frames of a LAMMPS custom dump (orthogonal boxes)."""
    with open(path) as fh:
        index = 0
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()
            n = int(fh.readline())
            fh.readline()
            bounds = np.array([fh.readline().split()[:2] for _ in range(3)], dtype=float)
            names = fh.readline().split()[2:]
            if index % every:
                for _ in range(n):
                    fh.readline()
                index += 1
                continue
            if n:
                data = np.loadtxt([fh.readline() for _ in range(n)], ndmin=2)
            else:
                data = np.zeros((0, len(names)))
            cols = {}
            for i, name in enumerate(names):
                col = data[:, i]
                cols[name] = col.astype(int) if name in ("id", "mol", "type", "ix", "iy", "iz") else col
            yield Frame(step, bounds[:, 0], bounds[:, 1], cols)
            index += 1


def read_dump(path: str | Path, every: int = 1) -> list[Frame]:
    return list(iter_dump(path, every))
