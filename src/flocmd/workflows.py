"""Workflow registry: builder + LAMMPS input + presets + analysis, and the sweep engine.

A *run directory* is self-contained: ``data.lmp`` (if the workflow has a builder),
``system.json`` (builder metadata), ``params.json`` (every parameter) and
``lammps_vars.json`` (the -var arguments). It can be executed later, e.g. on a
cluster, with ``flocmd run <run_dir>``.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from . import analyze
from .builders import (ColloidPEParams, DPDInterfaceParams, build_colloid_pe, build_dpd_interface,
                       build_single_chain)
from .runner import run_lammps

ROOT = Path(os.environ.get("FLOCMD_ROOT", Path(__file__).resolve().parents[2]))
SIM = ROOT / "simulations"


@dataclass
class Workflow:
    name: str
    input: Path
    build: Callable[[dict], tuple] | None
    builder_keys: set[str]
    links: dict[str, str]                 # builder key -> LAMMPS variable that must match it
    presets: dict[str, dict]
    analyze: Callable[[Path], dict]
    defaults: dict = field(default_factory=dict)


def _keys(cls) -> set[str]:
    return {f.name for f in fields(cls)}


def _build_colloid(params: dict):
    return build_colloid_pe(ColloidPEParams(**params))


def _build_pmf(params: dict):
    p = {"n_colloids": 2, "box_length": 40.0, "n_chains": 0, **params}
    return build_colloid_pe(ColloidPEParams(**p))


def _build_interface(params: dict):
    p = dict(params)
    if "box" in p:
        p["box"] = tuple(float(v) for v in p["box"])
    return build_dpd_interface(DPDInterfaceParams(**p))


def _build_chain(params: dict):
    return build_single_chain(int(params["chain_length"]), int(params.get("seed", 1)))


WORKFLOWS: dict[str, Workflow] = {
    "flocculation": Workflow(
        name="flocculation",
        input=SIM / "01_flocculation_kinetics" / "in.flocculation.lmp",
        build=_build_colloid,
        builder_keys=_keys(ColloidPEParams),
        links={"colloid_diameter": "dc", "colloid_charge": "Zc", "seed": "seed"},
        defaults={"colloid_diameter": 6.0, "colloid_charge": 20.0},
        presets={
            "smoke": {"n_colloids": 8, "n_push": 1000, "n_adsorb": 1000, "n_run": 2000,
                      "n_thermo": 500, "n_dump": 1000, "n_rdf": 1000},
            "quick": {"n_colloids": 27, "n_push": 10000, "n_adsorb": 20000, "n_run": 200000,
                      "n_thermo": 2000, "n_dump": 5000, "n_rdf": 50000},
            "production": {"n_colloids": 64, "n_push": 20000, "n_adsorb": 50000, "n_run": 2000000,
                           "n_thermo": 5000, "n_dump": 20000, "n_rdf": 200000},
        },
        analyze=analyze.flocculation,
    ),
    "pmf": Workflow(
        name="pmf",
        input=SIM / "02_bridging_pmf" / "in.pmf.lmp",
        build=_build_pmf,
        builder_keys=_keys(ColloidPEParams),
        links={"colloid_diameter": "dc", "colloid_charge": "Zc", "colloid_separation": "r_start",
               "seed": "seed"},
        defaults={"colloid_diameter": 6.0, "colloid_charge": 20.0, "colloid_separation": 6.5},
        presets={
            "smoke": {"n_push": 500, "n_adsorb": 1000, "n_windows": 3, "n_win_eq": 500,
                      "n_win_prod": 2000, "n_thermo": 500},
            "quick": {"n_push": 10000, "n_adsorb": 50000, "n_windows": 20, "n_win_eq": 10000,
                      "n_win_prod": 100000, "n_thermo": 10000},
            "production": {"n_push": 20000, "n_adsorb": 100000, "n_windows": 26, "dr": 0.4,
                           "n_win_eq": 50000, "n_win_prod": 500000, "n_thermo": 50000},
        },
        analyze=analyze.pmf,
    ),
    "interface": Workflow(
        name="interface",
        input=SIM / "03_oil_water_interface" / "in.interface.lmp",
        build=_build_interface,
        builder_keys=_keys(DPDInterfaceParams),
        links={"seed": "seed"},
        presets={
            "smoke": {"box": [8.0, 8.0, 16.0], "n_chains": 4, "chain_length": 20, "n_eq": 500,
                      "n_run": 1000, "n_prof": 500, "n_dump": 250, "n_thermo": 250},
            "quick": {"box": [12.0, 12.0, 30.0], "n_chains": 12, "chain_length": 30, "n_eq": 20000,
                      "n_run": 50000, "n_prof": 10000, "n_dump": 2500, "n_thermo": 2500},
            "production": {"box": [20.0, 20.0, 40.0], "n_chains": 32, "chain_length": 30,
                           "n_eq": 50000, "n_run": 250000, "n_prof": 25000, "n_dump": 5000},
        },
        analyze=analyze.interface,
    ),
    "eos": Workflow(
        name="eos",
        input=SIM / "00_validation" / "in.dpd_eos.lmp",
        build=None,
        builder_keys=set(),
        links={},
        presets={
            "smoke": {"L": 6.0, "n_eq": 200, "n_run": 1000, "a_list": [25.0]},
            "quick": {"L": 10.0, "n_eq": 2000, "n_run": 20000},
            "production": {"L": 12.0, "n_eq": 5000, "n_run": 100000},
        },
        analyze=analyze.eos,
    ),
    "chain_scaling": Workflow(
        name="chain_scaling",
        input=SIM / "00_validation" / "in.chain_scaling.lmp",
        build=_build_chain,
        builder_keys={"chain_length", "seed"},
        links={"seed": "seed"},
        presets={
            "smoke": {"n_push": 1000, "n_eq": 1000, "n_run": 5000},
            "quick": {"n_push": 10000, "n_eq": 200000, "n_run": 2000000},
            "production": {"n_push": 10000, "n_eq": 1000000, "n_run": 20000000},
        },
        analyze=analyze.chain_scaling,
    ),
}


def get(name: str) -> Workflow:
    try:
        return WORKFLOWS[name]
    except KeyError:
        raise KeyError(f"unknown workflow {name!r}; choose from {sorted(WORKFLOWS)}") from None


# --------------------------------------------------------------------------- #
# Preparing and running a single run directory                                #
# --------------------------------------------------------------------------- #
def prepare(workflow: str, run_dir: str | Path, params: dict | None = None,
            preset: str | None = None) -> dict:
    wf = get(workflow)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    merged = {**wf.defaults, **(wf.presets.get(preset, {}) if preset else {}), **(params or {})}
    builder_args = {k: v for k, v in merged.items() if k in wf.builder_keys}
    lammps_vars = {k: v for k, v in merged.items() if k not in wf.builder_keys}
    if wf.build is not None:
        system, meta = wf.build(builder_args)
        system.write(run_dir / "data.lmp", title=f"flocmd {workflow}")
        (run_dir / "system.json").write_text(json.dumps(meta, indent=1, default=float))
        lammps_vars["data"] = "data.lmp"
    for bkey, lvar in wf.links.items():
        if bkey in merged:
            lammps_vars[lvar] = merged[bkey]
    if wf.input.read_text().find("${common}") >= 0:
        lammps_vars["common"] = (SIM / "common").as_posix()
    (run_dir / "params.json").write_text(json.dumps(
        {"workflow": workflow, "preset": preset, "params": merged}, indent=1, default=float))
    (run_dir / "lammps_vars.json").write_text(json.dumps(lammps_vars, indent=1, default=float))
    return lammps_vars


def run(run_dir: str | Path, force: bool = False) -> Path:
    run_dir = Path(run_dir)
    if (run_dir / "done.txt").exists() and not force:
        return run_dir
    info = json.loads((run_dir / "params.json").read_text())
    lammps_vars = json.loads((run_dir / "lammps_vars.json").read_text())
    run_lammps(get(info["workflow"]).input, run_dir, lammps_vars)
    (run_dir / "done.txt").write_text("ok\n")
    return run_dir


# --------------------------------------------------------------------------- #
# Sweeps                                                                       #
# --------------------------------------------------------------------------- #
def _tag(values: dict) -> str:
    def short(v):
        if isinstance(v, (list, tuple)):
            return "x".join(short(x) for x in v)
        return f"{v:g}" if isinstance(v, float) else str(v)
    tag = "_".join(f"{k}-{short(v)}" for k, v in values.items()) or "base"
    return re.sub(r"[^A-Za-z0-9_.\-]", "", tag)


def expand(spec: dict) -> list[tuple[str, dict]]:
    """Cartesian product of ``[grid]`` combined with optional ``[[cases]]``."""
    base = spec.get("params", {})
    grid = spec.get("grid", {})
    cases = spec.get("cases", [{}])
    keys = list(grid)
    out = []
    for case in cases:
        for combo in itertools.product(*(grid[k] for k in keys)):
            varying = {**case, **dict(zip(keys, combo))}
            label = varying.pop("label", None)
            tag = (f"{label}_" if label else "") + _tag(varying)
            out.append((tag, {**base, **varying}))
    return out


def load_sweep(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def sweep(path: str | Path, jobs: int = 1, preset: str | None = None, output: str | Path | None = None,
          dry_run: bool = False, force: bool = False, log=print) -> Path:
    spec = load_sweep(path)
    workflow = spec["workflow"]
    preset = preset or spec.get("preset")
    out = Path(output or spec.get("output", f"runs/{workflow}_{preset}"))
    if not out.is_absolute():
        out = Path.cwd() / out
    runs = expand(spec)
    out.mkdir(parents=True, exist_ok=True)
    dirs = []
    for tag, params in runs:
        d = out / tag
        if force or not (d / "params.json").exists():
            prepare(workflow, d, params, preset)
        dirs.append(d)
    (out / "jobs.txt").write_text("\n".join(str(d) for d in dirs) + "\n")
    log(f"[flocmd] {len(dirs)} runs of '{workflow}' (preset {preset}) prepared in {out}")
    if dry_run:
        return out
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(run, d, force): d for d in dirs}
        for k, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                fut.result()
                log(f"[flocmd] ({k}/{len(dirs)}) done   {d.name}")
            except Exception as exc:  # keep the sweep going; report at the end
                log(f"[flocmd] ({k}/{len(dirs)}) FAILED {d.name}: {exc}")
    return out
