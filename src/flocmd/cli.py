"""Command-line interface: ``flocmd <command> ...`` (see ``flocmd -h``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_sets(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        out[key.strip()] = _parse_value(value.strip())
    return out


def cmd_info(args) -> None:
    from . import theory, workflows
    from .runner import LammpsNotFound, lammps_command

    try:
        lmp = " ".join(lammps_command())
    except LammpsNotFound as exc:
        lmp = f"NOT FOUND ({exc})"
    print(f"LAMMPS command : {lmp}")
    print(f"simulations    : {workflows.SIM}")
    print(f"workflows      : {', '.join(sorted(workflows.WORKFLOWS))}")
    sigma = args.sigma_nm
    lB = theory.bjerrum_length_nm()
    print(f"\nUnit mapping (Modules 1–2), sigma = {sigma} nm, water at 25 C: l_B = {lB:.3f} nm = {lB / sigma:.2f} sigma")
    print("  Debye length (sigma)   (nm)    ionic strength (mM)")
    for deb in (0.5, 1.0, 2.0, 3.0, 4.0):
        print(f"  {deb:8.1f}           {deb * sigma:6.2f}   {theory.ionic_strength_from_debye_M(deb * sigma) * 1e3:10.1f}")
    print(f"\nDPD (Module 3): r_c = {theory.dpd_cutoff_nm():.3f} nm (rho = 3, 3 H2O per bead)")


def cmd_build(args) -> None:
    from .workflows import prepare

    vars_ = prepare(args.workflow, args.output, _parse_sets(args.set), args.preset)
    print(f"prepared {args.workflow} in {args.output}")
    print("LAMMPS variables:", json.dumps(vars_))


def cmd_run(args) -> None:
    from .workflows import prepare, run

    if args.workflow:
        prepare(args.workflow, args.run_dir, _parse_sets(args.set), args.preset)
    run(args.run_dir, force=args.force)
    print(f"finished {args.run_dir}")
    if args.analyze:
        cmd_analyze(argparse.Namespace(path=args.run_dir, reanalyze=True))


def cmd_sweep(args) -> None:
    from .workflows import sweep

    out = sweep(args.file, jobs=args.jobs, preset=args.preset, output=args.output,
                dry_run=args.dry_run, force=args.force)
    if not args.dry_run and args.analyze:
        cmd_analyze(argparse.Namespace(path=out, reanalyze=True))


def cmd_analyze(args) -> None:
    from . import analyze
    from .workflows import get

    path = Path(args.path)
    if (path / "params.json").exists():
        info = json.loads((path / "params.json").read_text())
        summary = get(info["workflow"]).analyze(path)
        print(json.dumps(summary, indent=1, default=float))
    else:
        rows = analyze.collect(path, reanalyze=args.reanalyze)
        print(f"analysed {len(rows)} runs -> {path / 'summary.csv'}")


def cmd_plot(args) -> None:
    from . import plotting

    p = Path(args.path)
    if args.workflow == "flocculation":
        # one figure per Debye length (and kinetics per charge fraction) unless one is chosen,
        # so runs that differ in f or salt are never averaged into the same curve
        from .analyze import read_csv

        d = read_csv(p / "summary.csv")
        debyes = [args.debye] if args.debye is not None else plotting._groups(d, "debye")
        fracs = [args.charge_fraction] if args.charge_fraction is not None else plotting._groups(d, "charge_fraction")
        for deb in debyes:
            sfx = f"_debye-{deb:g}" if len(debyes) > 1 else ""
            print(plotting.flocculation_dose_response(p, p / f"dose_response{sfx}.png", debye=deb))
            for f in fracs:
                print(plotting.flocculation_kinetics(
                    p, f, deb, p / f"kinetics{sfx}{f'_f-{f:g}' if len(fracs) > 1 else ''}.png"))
    elif args.workflow == "pmf":
        cases = []
        for d in p.glob("*/pmf.csv"):
            s = json.loads((d.parent / "summary.json").read_text())
            if not s["n_chains"]:
                label = "no polymer (WHAM)"
            else:
                way = "separating" if s["direction"] > 0 else "approaching"
                label = f"{s['n_chains']} chain{'s' * (s['n_chains'] > 1)}, Φ = {s['dose']:.2f}, {way}"
            cases.append(((s["n_chains"], -s["direction"]), d.parent, label))
        cases.sort(key=lambda c: c[0])
        print(plotting.pmf_curves([c[1] for c in cases], [c[2] for c in cases], p / "pmf.png"))
    elif args.workflow == "interface":
        print(plotting.interface_summary(p))
        for d in sorted(x.parent for x in p.glob("*/profiles.csv")):
            plotting.interface_profiles(d)
    elif args.workflow == "eos":
        print(plotting.eos_check(p))
    elif args.workflow == "chain_scaling":
        out, nu, err = plotting.chain_scaling_plot(p)
        print(out, f"nu = {nu:.3f} +- {err:.3f}")
    else:
        raise SystemExit(f"no plots for {args.workflow}")


def cmd_movie(args) -> None:
    from . import movie

    p = Path(args.run_dir)
    if (p / "movie.lammpstrj").exists():
        print(movie.interface_movie(p, every=args.every))
    else:
        print(movie.flocculation_movie(p, every=args.every))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="flocmd", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="show LAMMPS detection and the reduced-unit mapping")
    s.add_argument("--sigma-nm", type=float, default=0.71)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("build", help="create a run directory (data file + variables), do not run")
    s.add_argument("workflow")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--preset", default="quick")
    s.add_argument("--set", action="append", metavar="KEY=VALUE")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("run", help="run a prepared directory (or build+run with --workflow)")
    s.add_argument("run_dir")
    s.add_argument("--workflow")
    s.add_argument("--preset", default="quick")
    s.add_argument("--set", action="append", metavar="KEY=VALUE")
    s.add_argument("--force", action="store_true")
    s.add_argument("--analyze", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="prepare and run a parameter sweep from a TOML file")
    s.add_argument("file")
    s.add_argument("-j", "--jobs", type=int, default=1, help="concurrent LAMMPS runs")
    s.add_argument("--preset")
    s.add_argument("-o", "--output")
    s.add_argument("--dry-run", action="store_true", help="only prepare run directories + jobs.txt")
    s.add_argument("--force", action="store_true")
    s.add_argument("--analyze", action="store_true")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("analyze", help="analyse one run directory or every finished run of a sweep")
    s.add_argument("path")
    s.add_argument("--reanalyze", action="store_true")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("plot", help="make figures for an analysed sweep")
    s.add_argument("workflow")
    s.add_argument("path")
    s.add_argument("--charge-fraction", type=float)
    s.add_argument("--debye", type=float)
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("movie", help="render an animated GIF of a run")
    s.add_argument("run_dir")
    s.add_argument("--every", type=int, default=1)
    s.set_defaults(func=cmd_movie)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
