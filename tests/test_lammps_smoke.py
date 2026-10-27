"""End-to-end smoke tests: build -> LAMMPS (a few thousand steps) -> analysis.

Skipped automatically when no LAMMPS executable is available (see flocmd.runner).
"""

import json

import numpy as np
import pytest

from flocmd import workflows
from flocmd.runner import lammps_available

pytestmark = [pytest.mark.lammps,
              pytest.mark.skipif(not lammps_available(), reason="LAMMPS executable not found")]


def _run(workflow, tmp_path, params=None):
    d = tmp_path / workflow
    workflows.prepare(workflow, d, params or {}, preset="smoke")
    workflows.run(d)
    return d, workflows.get(workflow).analyze(d)


def test_flocculation_smoke(tmp_path):
    d, s = _run("flocculation", tmp_path, {"dose": 1.0})
    assert (d / "traj.lammpstrj").exists()
    assert 1 <= s["n_flocs"] <= 8
    assert 0.0 <= s["frac_chains_adsorbed"] <= 1.0
    assert s["frac_train"] + s["frac_loop"] + s["frac_tail"] == pytest.approx(1.0, abs=1e-6)


def test_pmf_smoke(tmp_path):
    d, s = _run("pmf", tmp_path, {"n_chains": 1})
    assert len(np.loadtxt(d / "windows.txt", ndmin=2)) == 3
    assert np.isfinite(s["pmf_min"])


def test_interface_smoke(tmp_path):
    d, s = _run("interface", tmp_path, {"architecture": "random", "hydrophobic_fraction": 0.2,
                                        "n_eq": 3000, "n_run": 3000, "n_dump": 500})
    assert s["ift"] > 0
    assert 0 < s["interface_width"] < 5
    assert s["max_bond_length"] < 1.5       # image flags stay consistent (no wrong-image bonds)


def test_eos_smoke(tmp_path):
    d, s = _run("eos", tmp_path)
    assert 0.08 < s["alpha_mean"] < 0.11


def test_chain_scaling_smoke(tmp_path):
    d, s = _run("chain_scaling", tmp_path, {"chain_length": 16})
    params = json.loads((d / "params.json").read_text())
    assert params["params"]["chain_length"] == 16
    assert 1.0 < s["rg"] < 4.0
