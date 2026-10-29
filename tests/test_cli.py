import json

import numpy as np
import pytest

from flocmd import cli, runner
from flocmd.plotting import DOSE_PANELS


# ------------------------------------------------------------------ runner ---
def test_lammps_command_accepts_a_path_with_spaces(tmp_path, monkeypatch):
    exe = tmp_path / "LAMMPS 64-bit 22Jul2025" / "bin" / "lmp.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    for value in (str(exe), f'"{exe}"'):           # as the Windows docs write it, bare or quoted
        monkeypatch.setenv("FLOCMD_LMP", value)
        assert runner.lammps_command() == [str(exe)]
    monkeypatch.setenv("FLOCMD_LMP", "mpirun -np 4 lmp")
    assert runner.lammps_command() == ["mpirun", "-np", "4", "lmp"]


# -------------------------------------------------------------------- plot ---
def _fake_flocculation_sweep(root, fractions=(0.25, 1.0), doses=(0.5, 1.0)):
    """A sweep directory with the files `flocmd plot flocculation` reads (no LAMMPS needed)."""
    rows = []
    for f in fractions:
        for dose in doses:
            run = f"charge_fraction-{f:g}_dose-{dose:g}"
            d = root / run
            d.mkdir()
            (d / "params.json").write_text(json.dumps({"params": {"dt": 0.01, "gap_clus": 2.0}}))
            (d / "system.json").write_text(json.dumps(
                {"n_colloids": 16, "box_length": 60.0, "params": {"colloid_diameter": 6.0}}))
            t = np.arange(0, 2001, 100)
            n = 16 / (1 + t / (300 * f * dose))
            (d / "clusters.dat").write_text("# TimeStep v_nclus v_smean\n"
                                            + "\n".join(f"{a} {b} {16 / b}" for a, b in zip(t, n)))
            rows.append({"run": run, "dose": dose, "charge_fraction": f, "debye": 2.0,
                         **{key: 1.0 for key, *_ in DOSE_PANELS}})
    header = list(rows[0])
    (root / "summary.csv").write_text(",".join(header) + "\n"
                                      + "\n".join(",".join(str(r[k]) for k in header) for r in rows))


def test_plot_flocculation_never_mixes_charge_fractions(tmp_path):
    pytest.importorskip("matplotlib")
    _fake_flocculation_sweep(tmp_path)
    cli.main(["plot", "flocculation", str(tmp_path)])
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "dose_response.png", "kinetics_f-0.25.png", "kinetics_f-1.png"}
    cli.main(["plot", "flocculation", str(tmp_path), "--charge-fraction", "1"])
    assert (tmp_path / "kinetics.png").exists()
