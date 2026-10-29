"""Publication-style figures for every workflow (matplotlib, light theme).

Conventions: categorical hues in a fixed, colour-vision-deficiency-checked
order; an ordered blue ramp for ordered variables (dose); 2 px lines, >= 8 px
markers, hairline solid grid, text in ink colours, one y-axis per panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import theory  # noqa: E402
from .analyze import read_csv  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # ordinal
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.markersize": 6.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.labelcolor": INK2, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _line(ax, x, y, color, label=None, err=None, marker="o"):
    ax.plot(x, y, color=color, marker=marker, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, solid_joinstyle="round")
    if err is not None:
        ax.fill_between(x, np.asarray(y) - err, np.asarray(y) + err, color=color, alpha=0.10, lw=0)


def _groups(rows: dict, key: str) -> list:
    return sorted(set(rows[key].tolist())) if key in rows else [None]


# --------------------------------------------------------------------------- #
DOSE_PANELS = [
    ("contact_size_weight", "(a) Compact flocs: direct contacts", "Weight-avg. size (colloids)", False),
    ("floc_size_weight", "(b) Bridged flocs: contacts + polymer bridges", "Weight-avg. size (colloids)", False),
    ("net_charge_per_colloid", "(c) Net floc charge", "Charge per colloid (e)", False),
    ("stability_ratio", "(d) Stability ratio of contact aggregation", r"$W = k_{fast}/k$", True),
    ("frac_train", "(e) Adsorbed conformation", "Fraction of monomers in trains", False),
    ("frac_chains_bridging", "(f) Bridging", "Fraction of chains bridging", False),
]


def flocculation_dose_response(sweep_dir, out: str | Path | None = None, debye: float | None = None) -> Path:
    """Six-panel dose response, one series per charge fraction (one Debye length per figure)."""
    style()
    sweep_dir = Path(sweep_dir)
    d = read_csv(sweep_dir / "summary.csv")
    debyes = _groups(d, "debye")
    deb = debye if debye is not None else debyes[0]
    base = (d["debye"] == deb) if deb is not None else np.ones(len(d["run"]), bool)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, (key, title, ylabel, logy) in zip(axes.flat, DOSE_PANELS):
        for k, f in enumerate(_groups(d, "charge_fraction")):
            sel = base & (d["charge_fraction"] == f)
            doses = np.unique(d["dose"][sel])
            vals = [d[key][sel & (d["dose"] == x)].astype(float) for x in doses]
            mean = np.array([np.nanmean(v) if np.isfinite(v).any() else np.nan for v in vals])
            sem = np.array([np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()) if np.isfinite(v).sum() > 1 else 0.0
                            for v in vals])
            ok = np.isfinite(mean)
            _line(ax, doses[ok], mean[ok], SERIES[k], f"charge fraction f = {f:g}", sem[ok])
        if key == "net_charge_per_colloid":
            ax.axhline(0, color=AXIS, lw=0.8)
        if logy:
            ax.set_yscale("log")
            ax.axhline(1, color=AXIS, lw=0.8)
        ax.axvline(1.0, color=AXIS, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Dose $\Phi$ = polymer charge / colloid charge")
    axes[0, 0].legend(loc="upper right")
    label = f"Debye length {deb:g} σ" if deb is not None else ""
    fig.suptitle(f"Dose response of charged nanocolloids to a cationic flocculant   {label}",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    out = Path(out or sweep_dir / "dose_response.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def flocculation_kinetics(sweep_dir, charge_fraction: float | None = None, debye: float | None = None,
                          out: str | Path | None = None) -> Path:
    """Contact-cluster count N(t)/N0 for every dose (ordered blue ramp) vs fast Smoluchowski kinetics."""
    import json

    from .io import read_ave_time

    style()
    sweep_dir = Path(sweep_dir)
    d = read_csv(sweep_dir / "summary.csv")
    sel = np.ones(len(d["run"]), bool)
    if charge_fraction is not None:
        sel &= d["charge_fraction"] == charge_fraction
    if debye is not None and "debye" in d:
        sel &= d["debye"] == debye
    runs = sorted(zip(d["dose"][sel], d["run"][sel]))
    doses = sorted(set(x for x, _ in runs))
    ramp = BLUE_RAMP if len(doses) <= len(BLUE_RAMP) else plt.cm.Blues(np.linspace(0.35, 1, len(doses)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tmax = 0.0
    for k, dose in enumerate(doses):
        curves = []
        for _, run in [r for r in runs if r[0] == dose]:
            params = json.loads((sweep_dir / run / "params.json").read_text())["params"]
            meta = json.loads((sweep_dir / run / "system.json").read_text())
            c = read_ave_time(sweep_dir / run / "clusters.dat")
            curves.append((c["TimeStep"] * float(params.get("dt", 0.01)), c["v_nclus"] / meta["n_colloids"]))
        n = min(len(c[0]) for c in curves)
        t = curves[0][0][:n]
        tmax = max(tmax, t[-1])
        ax.plot(t, np.mean([c[1][:n] for c in curves], axis=0), color=ramp[k], label=f"$\\Phi$ = {dose:.2g}")
    # diffusion-limited reference: N/N0 = 1/(1 + t/tau_fast), tau_fast = 2/(k_fast n0), k_fast = 16 pi D a,
    # with the capture radius a = (dc + gap)/2 at which clusters.dat counts colloids as touching
    meta = json.loads((sweep_dir / runs[0][1] / "system.json").read_text())
    params = json.loads((sweep_dir / runs[0][1] / "params.json").read_text())["params"]
    dc = meta["params"]["colloid_diameter"]
    gap = float(params.get("gap_clus", 2.0))
    n0 = meta["n_colloids"] / meta["box_length"] ** 3
    tau_fast = 2.0 / (theory.smoluchowski_fast_rate(1.0 / dc, 0.5 * (dc + gap)) * n0)
    tt = np.linspace(0, tmax, 200)
    ax.plot(tt, 1.0 / (1.0 + tt / tau_fast), color=INK2, lw=1.0, label="fast Smoluchowski (W = 1)")
    ax.set_xlabel(r"Time after rapid mix ($\tau$)")
    ax.set_ylabel(r"Clusters of touching colloids, $N(t)/N_0$")
    ttl = f"Aggregation kinetics, f = {charge_fraction:g}" if charge_fraction is not None else "Aggregation kinetics"
    ax.set_title(ttl)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    out = Path(out or sweep_dir / "kinetics.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def pmf_curves(run_dirs: list, labels: list[str] | None = None, out: str | Path = "pmf.png",
               show_theory: bool = True) -> Path:
    """PMF w(r) with bootstrap bands; the bare DLVO + WCA potential as a reference line."""
    style()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ymax = 0.0
    for k, run in enumerate(run_dirs):
        d = read_csv(Path(run) / "pmf.csv")
        ok = np.isfinite(d["pmf"])
        label = labels[k] if labels else Path(run).name
        _line(ax, d["r"][ok], d["pmf"][ok], SERIES[k], label, d["pmf_err"][ok], marker="")
        ymax = max(ymax, np.nanpercentile(d["pmf"][ok], 95))
        if show_theory and k == 0:
            u = d["bare_pair_potential"]
            okt = np.isfinite(u) & (u < 1.5 * ymax + 5)
            ax.plot(d["r"][okt], u[okt], color=INK2, lw=1.0, label="no polymer, analytic DLVO + WCA")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel(r"Colloid–colloid distance $r$ ($\sigma$)")
    ax.set_ylabel(r"PMF $w(r)$ ($k_BT$)")
    ax.set_title("Bridging free energy (umbrella sampling + WHAM)")
    ax.legend(loc="best")
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


ARCH_ORDER = ["homopolymer", "random", "blocky", "telechelic", "diblock"]


def interface_summary(sweep_dir, out: str | Path | None = None) -> Path:
    """IFT change vs the polymer-free interface, adsorption and flattening for every architecture."""
    style()
    sweep_dir = Path(sweep_dir)
    d = read_csv(sweep_dir / "summary.csv")
    clean = d["n_chains"] == 0
    g0, g0_err = (d["ift"][clean].mean(), d["ift_err"][clean].mean()) if clean.any() else (0.0, 0.0)
    idx = [i for i in np.flatnonzero(~clean)]
    idx.sort(key=lambda i: (ARCH_ORDER.index(d["architecture"][i]) if d["architecture"][i] in ARCH_ORDER else 99,
                            d["hydrophobic_fraction"][i]))
    labels = ["PAM (no stickers)" if d["architecture"][i] == "homopolymer"
              else f"HM-PAM {d['architecture'][i]}, $f_H$ = {d['hydrophobic_fraction'][i]:.1f}" for i in idx]
    y = np.arange(len(idx))[::-1]
    ref = f"\n(clean interface: {g0:.2f} ± {g0_err:.2f})" if clean.any() else ""
    panels = [
        (d["ift"][idx] - g0, np.sqrt(d["ift_err"][idx] ** 2 + g0_err**2),
         r"$\Delta\gamma = \gamma - \gamma_{\mathrm{clean}}$ ($k_BT/r_c^2$)" + ref,
         "(a) Interfacial tension change", "{:+.2f}"),
        (d["frac_chains_adsorbed"][idx], None, "Fraction of chains at the interface", "(b) Adsorption", "{:.2f}"),
        (d["anisotropy_adsorbed"][idx], None, r"$\sqrt{R_{g\parallel}^2 / 2R_{g\perp}^2}$ of adsorbed chains",
         "(c) Flattening (1 = isotropic coil)", "{:.2f}"),
    ]
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(14, 0.5 * len(idx) + 2.0), sharey=True)
    for ax, (vals, err, xlabel, title, fmt) in zip(axes, panels):
        vals = np.nan_to_num(vals.astype(float), nan=0.0)
        ax.set_axisbelow(True)
        ax.barh(y, vals, height=0.5, color=SERIES[0], xerr=err, error_kw={"ecolor": INK2, "elinewidth": 0.8})
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ends = vals + np.sign(vals + 1e-12) * (err if err is not None else 0.0)
        lo, hi = min(0.0, ends.min()), max(0.0, ends.max())
        span = hi - lo or 1.0
        ax.set_xlim(lo - (0.25 * span if lo < 0 else 0.0), hi + 0.25 * span)
        for yi, v, e in zip(y, vals, ends):
            ax.text(e + (0.03 * span if v >= 0 else -0.03 * span), yi, fmt.format(v), va="center",
                    ha="left" if v >= 0 else "right", color=INK2, fontsize=8)
    axes[0].set_yticks(y, labels)
    fig.tight_layout()
    out = Path(out or sweep_dir / "interface_summary.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def interface_profiles(run_dir, out: str | Path | None = None) -> Path:
    style()
    run_dir = Path(run_dir)
    d = read_csv(run_dir / "profiles.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    names = [("water", "water (W)"), ("oil", "oil (O)"), ("backbone", "backbone (P)"), ("sticker", "sticker (H)")]
    for k, (key, label) in enumerate(names):
        y = d[key] if key in ("water", "oil") else 10 * d[key]
        ax.plot(d["z"], y, color=SERIES[k], label=label + ("" if key in ("water", "oil") else r" $\times$10"))
    ax.set_xlabel(r"$z$ ($r_c$)")
    ax.set_ylabel(r"Number density ($r_c^{-3}$)")
    ax.set_title("Density profiles across the oil slab")
    ax.legend(loc="upper right", fontsize=8)
    out = Path(out or run_dir / "profiles.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def eos_check(sweep_dir, out: str | Path | None = None) -> Path:
    style()
    sweep_dir = Path(sweep_dir)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    rows = sorted(sweep_dir.glob("*/eos.csv"))
    for k, f in enumerate(rows):
        d = read_csv(f)
        _line(ax, d["a"], d["alpha"], SERIES[k], f"$\\rho$ = {d['rho'][0]:g}")
    ax.axhline(0.101, color=INK2, lw=1.0)
    ax.text(ax.get_xlim()[1], 0.1008, "Groot–Warren fit, α = 0.101  ", ha="right", va="top", color=INK2,
            fontsize=8)
    ax.set_ylim(top=0.1025)
    ax.set_xlabel(r"Repulsion parameter $a$")
    ax.set_ylabel(r"$\alpha = (p-\rho k_BT)/(a\rho^2)$")
    ax.set_title("DPD equation of state: α is independent of a, rises with ρ")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    out = Path(out or sweep_dir / "eos.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def chain_scaling_plot(sweep_dir, out: str | Path | None = None, n_min: int = 32) -> tuple[Path, float, float]:
    """Rg vs N; nu is fitted for N >= n_min (short chains carry corrections to scaling)."""
    from .analysis.polymer import flory_exponent

    style()
    sweep_dir = Path(sweep_dir)
    d = read_csv(sweep_dir / "summary.csv")
    n = np.unique(d["chain_length"])
    rg = np.array([d["rg"][d["chain_length"] == x].mean() for x in n])
    err = np.array([np.sqrt(np.sum(d["rg_err"][d["chain_length"] == x] ** 2)) / (d["chain_length"] == x).sum()
                    for x in n])
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    use = n >= n_min
    nu, nu_err, b = flory_exponent(n[use], rg[use], err[use])
    local = np.diff(np.log(rg)) / np.diff(np.log(n))
    rel = err / rg
    local_err = np.sqrt(rel[1:] ** 2 + rel[:-1] ** 2) / np.diff(np.log(n))
    print("local exponents:", ", ".join(f"{a:g}-{c:g}: {v:.3f}±{e:.3f}"
                                        for a, c, v, e in zip(n[:-1], n[1:], local, local_err)))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax.errorbar(n, rg, yerr=err, fmt="o", color=SERIES[0], mec=SURFACE, mew=1.5, ecolor=INK2, elinewidth=0.8)
    xx = np.geomspace(n.min() * 0.8, n.max() * 1.2, 50)
    ax.plot(xx, b * xx**nu, color=SERIES[0], lw=1.2, label=f"fit N ≥ {n_min}: ν = {nu:.3f} ± {nu_err:.3f}")
    ax.plot(xx, b * float(n[use].min()) ** (nu - 0.588) * xx**0.588, color=INK2, lw=1.0, label="ν = 0.588")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis, ticks in ((ax.xaxis, n), (ax.yaxis, [2, 3, 4, 6, 8, 12])):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks(n)
    ax.set_yticks([2, 3, 4, 6, 8, 12])
    ax.legend(loc="upper left")
    ax.set_xlabel("Chain length $N$")
    ax.set_ylabel(r"$R_g$ ($\sigma$)")
    ax.set_title("(a) Radius of gyration, good solvent")

    mid = np.sqrt(n[1:] * n[:-1])
    ax2.errorbar(mid, local, yerr=local_err, fmt="o-", color=SERIES[0], mec=SURFACE, mew=1.5,
                 ecolor=INK2, elinewidth=0.8)
    ax2.axhline(0.588, color=INK2, lw=1.0)
    ax2.text(mid[0], 0.588, "  ν = 0.588 (asymptotic)", va="bottom", color=INK2, fontsize=8)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xticks(np.round(mid).astype(int))
    ax2.set_xlabel("Chain length $N$ (geometric mid-point)")
    ax2.set_ylabel(r"local exponent $d\ln R_g / d\ln N$")
    ax2.set_title("(b) Local exponent: corrections to scaling at small N")
    fig.tight_layout()
    out = Path(out or sweep_dir / "chain_scaling.png")
    fig.savefig(out)
    plt.close(fig)
    return out, nu, nu_err


def dlvo_reference(out: str | Path, dc: float = 6.0, zc: float = 20.0, debyes=(1.0, 2.0, 4.0)) -> Path:
    """Bare colloid pair potential for several Debye lengths (salt screening)."""
    style()
    r = np.linspace(dc + 0.05, dc + 14, 400)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for k, deb in enumerate(debyes):
        ax.plot(r, theory.colloid_pair_potential(r, dc, zc, deb), color=SERIES[k],
                label=f"Debye length {deb:g} $\\sigma$ ({theory.ionic_strength_from_debye_M(0.71 * deb) * 1e3:.0f} mM)")
    ax.set_ylim(-1, 40)
    ax.set_xlabel(r"$r$ ($\sigma$)")
    ax.set_ylabel(r"$U(r)$ ($k_BT$)")
    ax.set_title("Bare colloid–colloid repulsion")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return Path(out)
