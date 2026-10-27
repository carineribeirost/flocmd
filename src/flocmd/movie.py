"""Lightweight trajectory movies (animated GIF) without OVITO/VMD.

Projections are rendered with matplotlib, so an animation can be produced on
any machine. For publication-quality rendering, load the same
dump files in OVITO (https://www.ovito.org).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .io import read_dump  # noqa: E402
from .plotting import INK2, SERIES, SURFACE, style  # noqa: E402


def flocculation_movie(run_dir, out=None, every: int = 1, fps: int = 8, dt: float = 0.01) -> Path:
    """Colloids (grey discs) and polyelectrolyte (blue = cationic, light = neutral), 3D view."""
    style()
    run_dir = Path(run_dir)
    frames = read_dump(run_dir / "traj.lammpstrj", every)
    L = frames[0].box_lengths
    params = json.loads((run_dir / "params.json").read_text())["params"]
    dc = float(params.get("colloid_diameter", 6.0))
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_subplot(projection="3d")
    pt_per_sigma = 0.6 * 72 * 5 / L[0]          # marker sizes follow the physical diameters

    def draw(k):
        ax.cla()
        f = frames[k]
        x = f.positions("x") - f.box_lo
        t = f["type"]
        ax.scatter(*x[t == 1].T, s=(dc * pt_per_sigma) ** 2, c="#b9b8b0", edgecolors="#898781",
                   linewidths=0.5, depthshade=True)
        ax.scatter(*x[t == 2].T, s=(0.6 * pt_per_sigma) ** 2, c="#9ec5f4", depthshade=False)
        ax.scatter(*x[t == 3].T, s=(0.8 * pt_per_sigma) ** 2, c=SERIES[0], depthshade=False)
        ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_zlim(0, L[2])
        ax.view_init(elev=18, azim=30 + 90 * k / max(1, len(frames) - 1))
        ax.set_axis_off()
        ax.text2D(0.04, 0.95, f"t = {f.timestep * dt:.0f} τ", transform=ax.transAxes, color=INK2)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    out = Path(out or run_dir / "flocculation.gif")
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def interface_movie(run_dir, out=None, every: int = 1, fps: int = 10, dt: float = 0.04) -> Path:
    """x–z projection of oil (light) and flocculant beads (P blue, H orange) from movie.lammpstrj."""
    style()
    run_dir = Path(run_dir)
    frames = read_dump(run_dir / "movie.lammpstrj", every)
    L = frames[0].box_lengths
    fig, ax = plt.subplots(figsize=(3.2, 5.6))

    def draw(k):
        ax.cla()
        f = frames[k]
        x = f.positions("x") - f.box_lo
        t = f["type"]
        ax.scatter(x[t == 2, 0], x[t == 2, 2], s=1, c="#f2d9a6", rasterized=True)
        ax.scatter(x[t == 3, 0], x[t == 3, 2], s=5, c=SERIES[0])
        ax.scatter(x[t == 4, 0], x[t == 4, 2], s=7, c=SERIES[1])
        ax.set_xlim(0, L[0]); ax.set_ylim(0, L[2])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_facecolor("#eef5fd")   # water background
        ax.set_title(f"t = {f.timestep * dt:.0f} τ", color=INK2, loc="left", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    out = Path(out or run_dir / "interface.gif")
    anim.save(out, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return out
