import numpy as np
import pytest

from flocmd.builders import (ColloidPEParams, DPDInterfaceParams, build_colloid_pe, build_dpd_interface,
                             build_single_chain)
from flocmd.builders.colloid_pe import charge_sequence, n_chains_for_dose
from flocmd.builders.dpd_interface import sticker_sequence
from flocmd.lammps_data import min_image


def test_dose_sets_chain_count_and_charge_ratio():
    p = ColloidPEParams(n_colloids=10, colloid_charge=20, chain_length=40, charge_fraction=0.5, dose=1.5)
    system, meta = build_colloid_pe(p)
    assert meta["n_chains"] == n_chains_for_dose(p) == 15        # 1.5 * 10 * 20 / 20
    assert meta["dose_actual"] == pytest.approx(1.5)
    q = system.charges
    assert q[system.types == 1].sum() == pytest.approx(-200)
    assert q[system.types == 3].sum() == pytest.approx(300)


def test_colloids_do_not_overlap_and_chains_avoid_cores():
    p = ColloidPEParams(n_colloids=20, dose=1.0, seed=7)
    system, meta = build_colloid_pe(p)
    L = meta["box_length"]
    c = system.positions[system.types == 1]
    d = np.linalg.norm(min_image(c[:, None] - c[None], L), axis=-1)
    d[np.diag_indices(len(c))] = np.inf
    assert d.min() >= p.colloid_diameter + p.min_gap - 1e-9
    m = system.positions[system.types != 1]
    dm = np.linalg.norm(min_image(m[:, None] - c[None], L), axis=-1)
    assert dm.min() >= 0.5 * p.colloid_diameter + 0.5


def test_bond_lengths_and_topology():
    system, meta = build_colloid_pe(ColloidPEParams(n_colloids=4, dose=1.0))
    b = system.bonds
    assert len(b) == meta["n_chains"] * (60 - 1)
    lengths = np.linalg.norm(system.positions[b[:, 2]] - system.positions[b[:, 1]], axis=1)
    assert np.allclose(lengths, 0.97)
    assert np.all(system.mol[b[:, 1]] == system.mol[b[:, 2]])


def test_pmf_geometry():
    system, _ = build_colloid_pe(ColloidPEParams(n_colloids=2, colloid_separation=9.0, box_length=30,
                                                 n_chains=1))
    c = system.positions[:2]
    assert np.linalg.norm(c[1] - c[0]) == pytest.approx(9.0)


@pytest.mark.parametrize("pattern", ["regular", "random"])
def test_charge_sequence_fraction(pattern):
    s = charge_sequence(60, 0.25, pattern, np.random.default_rng(0))
    assert s.sum() == 15


@pytest.mark.parametrize("arch,expected", [("homopolymer", 0), ("random", 6), ("blocky", 6),
                                           ("telechelic", 6), ("diblock", 6)])
def test_sticker_sequences(arch, expected):
    s = sticker_sequence(30, arch, 0.2, 3, np.random.default_rng(1))
    assert s.sum() == expected
    if arch == "diblock":
        assert s[:6].all()
    if arch == "telechelic":
        assert s[:3].all() and s[-3:].all()


def test_dpd_slab_composition():
    p = DPDInterfaceParams(box=(8.0, 8.0, 16.0), n_chains=4, chain_length=20)
    system, meta = build_dpd_interface(p)
    assert system.n_atoms == round(3.0 * 8 * 8 * 16)
    z1, z2 = meta["slab_z"]
    zo = system.positions[system.types == 2][:, 2] % 16.0
    assert np.all((zo >= z1) & (zo <= z2))
    zw = system.positions[system.types == 1][:, 2] % 16.0
    assert not np.any((zw > z1) & (zw < z2))


def test_data_file_roundtrip(tmp_path):
    system, _ = build_single_chain(16)
    path = system.write(tmp_path / "data.lmp")
    text = path.read_text()
    assert "16 atoms" in text and "15 bonds" in text and "Atoms # full" in text
    wrapped, image = system.wrapped()
    assert np.allclose(wrapped + image * system.box_lengths, system.positions)
    assert np.all((wrapped >= 0) & (wrapped < system.box_lengths))
