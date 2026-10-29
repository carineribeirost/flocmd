"""Per-run analysis for every workflow, plus sweep-level collection.

Each ``<workflow>(run_dir)`` function reads the raw LAMMPS output of one run,
writes tidy files into the run directory (``summary.json`` and CSV tables)
and returns the summary dict.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import theory
from .analysis import bridging, interface as iface, polymer, wham as whamlib
from .analysis.clusters import fit_smoluchowski
from .analysis.stats import block_average
from .io import iter_dump, read_ave_blocks, read_ave_time


def _load(run_dir: Path) -> tuple[dict, dict]:
    params = json.loads((run_dir / "params.json").read_text())["params"]
    meta_path = run_dir / "system.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return params, meta


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: str | Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def _save(run_dir: Path, summary: dict) -> dict:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


# --------------------------------------------------------------------------- #
def flocculation(run_dir, every: int = 1, late_fraction: float = 0.5) -> dict:
    """Floc size, bridging, train/loop/tail and floc charge vs time; Smoluchowski fit."""
    run_dir = Path(run_dir)
    params, meta = _load(run_dir)
    dt = float(params.get("dt", 0.01))
    gap = float(params.get("gap_clus", 2.0))
    rows = []
    for frame in iter_dump(run_dir / "traj.lammpstrj", every):
        r = bridging.analyze_frame(frame, meta, gap=gap)
        r["time"] = frame.timestep * dt
        rows.append(r)
    _write_csv(run_dir / "timeseries.csv", rows)

    n0 = int(meta["n_colloids"])
    volume = float(meta["box_length"]) ** 3
    dc = float(meta["params"]["colloid_diameter"])
    cl = read_ave_time(run_dir / "clusters.dat")
    t = cl["TimeStep"] * dt
    n = cl["v_nclus"]
    # Smoluchowski fit of *contact* aggregation (direct colloid contacts). Bridged flocs,
    # whose colloids stay a few sigma apart, are captured by the floc graph instead.
    # clusters.dat counts colloids as touching below a centre distance dc + gap, so the
    # diffusion-limited reference uses that capture distance too (W = 1 without any barrier).
    k_fast = theory.smoluchowski_fast_rate(D=1.0 / dc, radius=0.5 * (dc + gap))  # D = kT / (dc * gamma_m)
    tau = tau_err = k_obs = float("nan")
    if n[-1] < 0.9 * n0:
        early = n >= max(2.0, 0.25 * n0)  # constant-kernel kinetics only holds early on
        tau, tau_err = fit_smoluchowski(t[early], n[early], n0)
        k_obs = 2.0 / ((n0 / volume) * tau)

    late = rows[int((1 - late_fraction) * len(rows)):] or rows
    keys = [k for k in rows[0] if k not in ("timestep", "time", "n_colloids", "n_chains")]
    summary = {k: float(np.mean([r[k] for r in late])) for k in keys}
    summary.update({
        "dose": meta.get("dose_actual"), "n_chains": meta["n_chains"], "n_colloids": n0,
        "charge_fraction": meta["params"]["charge_fraction"], "chain_length": meta["params"]["chain_length"],
        "debye": float(params.get("debye", 2.0)), "eps_ads": float(params.get("eps_ads", 0.0)),
        "tau_agg": tau, "tau_agg_err": tau_err, "k_agg": k_obs,
        "stability_ratio": k_fast / k_obs, "t_final": float(t[-1]),
    })
    return _save(run_dir, summary)


def pmf(run_dir, discard: float = 0.1, bin_width: float = 0.05, n_boot: int = 30) -> dict:
    """WHAM PMF along the colloid–colloid distance; theory check when there is no polymer."""
    run_dir = Path(run_dir)
    params, meta = _load(run_dir)
    win = np.loadtxt(run_dir / "windows.txt", comments="#", ndmin=2)
    samples = []
    for w in win[:, 0].astype(int):
        r = read_ave_time(run_dir / f"window_{w}.dat")["v_r"]
        samples.append(r[int(discard * len(r)):])
    lo = min(s.min() for s in samples)
    hi = max(s.max() for s in samples)
    edges = np.arange(lo, hi + bin_width, bin_width)
    res = whamlib.bootstrap_wham(samples, win[:, 1], win[:, 2], bins=edges, n_boot=n_boot)
    outside = (res.centers < win[:, 1].min() - 0.25) | (res.centers > win[:, 1].max())
    res.pmf[outside] = np.nan     # tails beyond the outermost windows are barely sampled
    finite = np.isfinite(res.pmf)
    tail = np.flatnonzero(finite)[-max(1, finite.sum() // 20):]
    res.pmf -= res.pmf[tail].mean()
    res.pmf_err = whamlib.spread(res.replicates, ref=tail)   # error bars with the same zero

    dc = float(params.get("colloid_diameter", 6.0))
    zc = float(params.get("colloid_charge", 20.0))
    debye = float(params.get("debye", 2.0))
    lB = float(params.get("lB", 1.0))
    u_ref = theory.colloid_pair_potential(res.centers, dc, zc, debye, lB)
    u_ref = u_ref - u_ref[tail].mean()
    rows = [{"r": float(r), "pmf": float(p), "pmf_err": float(e), "free_energy": float(f),
             "bare_pair_potential": float(u)}
            for r, p, e, f, u in zip(res.centers, res.pmf, res.pmf_err, res.free_energy, u_ref)]
    _write_csv(run_dir / "pmf.csv", rows)

    ok = np.isfinite(res.pmf)
    i_min = int(np.nanargmin(np.where(ok, res.pmf, np.nan)))
    summary = {
        "n_chains": meta.get("n_chains", 0),
        "dose": meta.get("dose_actual"),
        "charge_fraction": meta.get("params", {}).get("charge_fraction"),
        "debye": debye, "direction": int(params.get("direction", 1)),
        "pmf_min": float(res.pmf[i_min]), "r_min": float(res.centers[i_min]),
        "pmf_min_err": float(res.pmf_err[i_min]),
        "wham_iterations": res.iterations,
    }
    sel = ok & np.isfinite(u_ref) & (u_ref < 8.0)
    if meta.get("n_chains", 0) == 0 and sel.any():
        summary["rmsd_vs_theory"] = float(np.sqrt(np.mean((res.pmf[sel] - u_ref[sel]) ** 2)))
    return _save(run_dir, summary)


def interface(run_dir, discard: float = 0.0, adsorption_cut: float = 1.0) -> dict:
    """Interfacial tension, density profiles, adsorption and chain anisotropy."""
    run_dir = Path(run_dir)
    params, meta = _load(run_dir)
    box = np.asarray(meta["box"], dtype=float)

    ift = read_ave_time(run_dir / "ift.dat")
    g = ift["v_ift"][int(discard * len(ift["v_ift"])):]
    gb = block_average(g)

    prof = {}
    for name in ("water", "oil", "backbone", "sticker"):
        z, rho, sem = iface.mean_profile(read_ave_blocks(run_dir / f"prof_{name}.dat"), discard)
        prof["z"], prof[name], prof[f"{name}_sem"] = z, rho, sem
    fit = iface.fit_slab(prof["z"], prof["oil"])
    _write_csv(run_dir / "profiles.csv",
               [{k: float(v[i]) for k, v in prof.items()} for i in range(len(prof["z"]))])

    first_poly = int(meta["first_polymer_mol"])
    ads_frac, aniso_ads, aniso_free, stick_depth, rg2 = [], [], [], [], []
    max_bond = 0.0
    for frame in iter_dump(run_dir / "polymer.lammpstrj"):
        order = np.argsort(frame["id"])
        mol = frame["mol"][order]
        sel = mol >= first_poly
        x = frame.positions("xu")[order][sel]
        m, typ = mol[sel], frame["type"][order][sel]
        if len(m) == 0:
            continue
        # integrity check: consecutive beads of a chain must stay bonded in unwrapped space
        same = m[1:] == m[:-1]
        max_bond = max(max_bond, float(np.linalg.norm(np.diff(x, axis=0), axis=1)[same].max()))
        com, S = polymer.gyration_tensors(x, m)
        comp = polymer.rg_components(S)
        d = iface.distance_to_interfaces(x[:, 2], fit["z1"], fit["z2"], box[2])
        adsorbed = np.array([d[m == k].min() <= adsorption_cut for k in np.unique(m)])
        ads_frac.append(adsorbed.mean())
        rg2.append(comp["rg2"].mean())
        if adsorbed.any():
            aniso_ads.append(comp["anisotropy"][adsorbed].mean())
        if (~adsorbed).any():
            aniso_free.append(comp["anisotropy"][~adsorbed].mean())
        if (typ == 4).any():
            stick_depth.append(d[typ == 4].mean())

    summary = {
        "architecture": meta["params"]["architecture"],
        "hydrophobic_fraction": meta["hydrophobic_fraction_actual"],
        "n_chains": meta["n_chains"], "chain_length": meta["params"]["chain_length"],
        "chains_per_area": meta["n_chains"] / (2 * box[0] * box[1]),
        "ift": gb.mean, "ift_err": gb.sem,
        "interface_width": fit["width"], "z1": fit["z1"], "z2": fit["z2"],
        "frac_chains_adsorbed": float(np.mean(ads_frac)) if ads_frac else 0.0,
        "anisotropy_adsorbed": float(np.mean(aniso_ads)) if aniso_ads else float("nan"),
        "anisotropy_free": float(np.mean(aniso_free)) if aniso_free else float("nan"),
        "rg2": float(np.mean(rg2)) if rg2 else float("nan"),
        "sticker_mean_distance": float(np.mean(stick_depth)) if stick_depth else float("nan"),
        "max_bond_length": max_bond,
    }
    if max_bond > 1.5:
        print(f"[flocmd] WARNING {run_dir.name}: unwrapped bond of {max_bond:.1f} r_c -> inconsistent image "
              "flags; conformational results are unreliable (see docs/science.md, DPD pitfall)")
    return _save(run_dir, summary)


def eos(run_dir) -> dict:
    """Groot–Warren scaling check: alpha = (p - rho kT) / (a rho^2) should not depend on a."""
    run_dir = Path(run_dir)
    d = np.loadtxt(run_dir / "eos.dat", comments="#", ndmin=2)
    a, rho, p, temp = d[:, 0], d[:, 1], d[:, 2], d[:, 4]
    alpha = (p - rho * temp) / (a * rho**2)
    rows = [{"a": float(x), "rho": float(r), "p": float(pp), "T": float(t), "alpha": float(al),
             "p_groot_warren": float(theory.groot_warren_pressure(r, x))}
            for x, r, pp, t, al in zip(a, rho, p, temp, alpha)]
    _write_csv(run_dir / "eos.csv", rows)
    return _save(run_dir, {"rho": float(rho[0]), "alpha_mean": float(alpha.mean()),
                           "alpha_spread": float(alpha.max() - alpha.min()), "T_mean": float(temp.mean())})


def chain_scaling(run_dir) -> dict:
    run_dir = Path(run_dir)
    params, _ = _load(run_dir)
    rg2 = read_ave_time(run_dir / "rg.dat")["v_rg2"]
    b = block_average(rg2)
    return _save(run_dir, {"chain_length": int(params["chain_length"]), "rg2": b.mean, "rg2_err": b.sem,
                           "rg": float(np.sqrt(b.mean)), "rg_err": float(0.5 * b.sem / np.sqrt(b.mean))})


# --------------------------------------------------------------------------- #
def collect(sweep_dir, analyzer=None, reanalyze: bool = False, log=print) -> list[dict]:
    """Analyse every finished run of a sweep and write ``<sweep_dir>/summary.csv``."""
    from .workflows import get

    sweep_dir = Path(sweep_dir)
    rows = []
    for run_dir in sorted(p.parent for p in sweep_dir.glob("*/done.txt")):
        info = json.loads((run_dir / "params.json").read_text())
        fn = analyzer or get(info["workflow"]).analyze
        try:
            if reanalyze or not (run_dir / "summary.json").exists():
                s = fn(run_dir)
            else:
                s = json.loads((run_dir / "summary.json").read_text())
        except Exception as exc:
            log(f"[flocmd] analysis failed for {run_dir.name}: {exc}")
            continue
        rows.append({"run": run_dir.name, **{k: v for k, v in s.items() if not isinstance(v, (list, dict))}})
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(sweep_dir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows
