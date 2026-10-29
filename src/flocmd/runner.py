"""Locate and launch LAMMPS.

Resolution order for the command prefix:
1. ``FLOCMD_LMP`` environment variable, e.g. ``"mpirun -np 4 lmp"`` or ``"lmp -sf omp -pk omp 4"``
2. the first of ``lmp``, ``lmp_serial``, ``lmp_mpi``, ``lmp.exe`` found on PATH
   (``pip install lammps`` provides ``lmp``; the Windows installer provides ``lmp.exe``)
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi", "lmp.exe")


class LammpsNotFound(RuntimeError):
    pass


def lammps_command() -> list[str]:
    env = os.environ.get("FLOCMD_LMP", "").strip()
    if env:
        bare = env.strip('"')
        if os.path.isfile(bare):   # a bare path to the executable, possibly with spaces (Windows)
            return [bare]
        # Windows (posix=False) keeps the quotes of a quoted path in its token: drop them
        return [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t
                for t in shlex.split(env, posix=os.name != "nt")]
    for name in CANDIDATES:
        path = shutil.which(name)
        if path:
            return [path]
    raise LammpsNotFound(
        "LAMMPS executable not found. Install it (`pip install lammps`, conda-forge, or the "
        "Windows installer) or set FLOCMD_LMP, e.g. FLOCMD_LMP='mpirun -np 4 lmp'."
    )


def lammps_available() -> bool:
    try:
        lammps_command()
        return True
    except LammpsNotFound:
        return False


def build_args(input_file: str | Path, variables: dict, log: str = "log.lammps") -> list[str]:
    args = ["-in", str(Path(input_file).resolve()), "-log", log, "-echo", "screen"]
    for key, value in variables.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        args += ["-var", str(key), *[_fmt(v) for v in values]]
    return args


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float) and v.is_integer() and abs(v) >= 1000:
        return str(int(v))
    return str(v)


def run_lammps(input_file: str | Path, cwd: str | Path, variables: dict | None = None,
               log: str = "log.lammps", screen: str = "screen.txt",
               command: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run LAMMPS in ``cwd``; stdout/stderr go to ``screen``. Raises on a non-zero exit code."""
    cwd = Path(cwd)
    cmd = (command or lammps_command()) + build_args(input_file, variables or {}, log)
    (cwd / "command.txt").write_text(shlex.join(cmd) + "\n")
    t0 = time.time()
    with open(cwd / screen, "w") as out:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, text=True)
    (cwd / "walltime.txt").write_text(f"{time.time() - t0:.1f}\n")
    if proc.returncode != 0:
        tail = (cwd / screen).read_text(errors="replace").splitlines()[-15:]
        raise RuntimeError(f"LAMMPS failed in {cwd} (exit {proc.returncode}):\n" + "\n".join(tail))
    return proc
