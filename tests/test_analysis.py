import numpy as np
import pytest

from flocmd import theory
from flocmd.analysis import bridging, clusters, interface, polymer, stats
from flocmd.analysis.wham import wham


# ---------------------------------------------------------------- clusters ---
def test_clusters_respect_periodic_boundaries():
    L = 10.0
    pos = np.array([[0.2, 5, 5], [9.8, 5, 5], [5, 5, 5], [5, 5, 6.5]])
    labels = clusters.find_clusters(pos, L, cutoff=1.0)
    assert labels[0] == labels[1]            # 0.4 apart through the boundary
    assert labels[2] != labels[3]            # 1.5 apart
    assert sorted(clusters.cluster_sizes(labels)) == [1, 1, 2]


def test_size_moments():
    m = clusters.size_moments(np.array([1, 1, 2]))
    assert m["s_number"] == pytest.approx(4 / 3)
    assert m["s_weight"] == pytest.approx(6 / 4)


def test_smoluchowski_fit_recovers_tau():
    t = np.linspace(0, 500, 60)
    n = theory.smoluchowski_n_clusters(t, n0=64, k=2.0, volume=64 / 0.01)   # tau = 2/(k n0) = 100
    tau, _ = clusters.fit_smoluchowski(t, n, 64)
    assert tau == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------- bridging ---
def test_train_loop_tail():
    ads = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0], bool)
    r = bridging.train_loop_tail(ads)
    assert (r["train"], r["loop"], r["tail"]) == (3, 3, 3)
    assert (r["n_trains"], r["n_loops"]) == (2, 1)
    assert bridging.train_loop_tail(np.zeros(5, bool))["train"] == 0


def test_adsorption_map_assigns_nearest_colloid():
    colloids = np.array([[2.0, 5, 5], [8.0, 5, 5]])
    monomers = np.array([[5.4, 5, 5], [4.6, 5, 5], [9.9, 5, 5], [0.1, 5, 5]])
    owner = bridging.adsorption_map(colloids, monomers, 10.0, r_ads=4.0)
    assert owner.tolist() == [1, 0, 1, 0]    # last two use the periodic image


# --------------------------------------------------------------- stats --------
def test_block_average_matches_ar1_theory():
    rng = np.random.default_rng(3)
    phi, n = 0.9, 2**17
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    var = 1 / (1 - phi**2)
    sem_exact = np.sqrt(var * (1 + phi) / (1 - phi) / n)
    r = stats.block_average(x)
    assert r.sem == pytest.approx(sem_exact, rel=0.25)
    assert r.sem > 3 * np.sqrt(x.var() / n)          # naive SEM is far too small
    assert stats.statistical_inefficiency(x) == pytest.approx((1 + phi) / (1 - phi), rel=0.25)


# --------------------------------------------------------------- interface ----
def test_fit_slab_recovers_parameters():
    z = np.linspace(0, 30, 121)
    rho = interface.slab_profile(z, 3.0, 0.05, 9.0, 21.0, 0.8)
    fit = interface.fit_slab(z, rho)
    assert fit["z1"] == pytest.approx(9.0, abs=1e-3)
    assert fit["z2"] == pytest.approx(21.0, abs=1e-3)
    assert fit["width"] == pytest.approx(0.8, abs=1e-3)


def test_distance_to_interfaces_sign_and_pbc():
    d = interface.distance_to_interfaces(np.array([15.0, 8.0, 1.0, 29.0]), 9.0, 21.0, 30.0)
    assert d[0] < 0 and d[1] > 0
    assert d[2] == pytest.approx(8.0) and d[3] == pytest.approx(8.0)


def test_interfacial_tension_formula():
    assert interface.interfacial_tension(1.0, 1.0, 3.0, 40.0) == pytest.approx(40.0)


# --------------------------------------------------------------- polymer ------
def test_rg_and_anisotropy_of_flat_ring():
    phi = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    x = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    _, S = polymer.gyration_tensors(x, np.ones(50, int))
    c = polymer.rg_components(S)
    assert c["rg2"][0] == pytest.approx(1.0)
    assert c["anisotropy"][0] > 100


def test_flory_exponent_fit():
    n = np.array([16, 32, 64, 128])
    nu, err, b = polymer.flory_exponent(n, 0.4 * n**0.588)
    assert nu == pytest.approx(0.588, abs=1e-9)


# --------------------------------------------------------------- WHAM ---------
def _sample_biased(pmf, grid, r0, k, n, rng):
    """Exact samples of r from p(r) ~ r^2 exp(-w(r) - k/2 (r - r0)^2) on a grid."""
    logp = 2 * np.log(grid) - pmf - 0.5 * k * (grid - r0) ** 2
    p = np.exp(logp - logp.max())
    cdf = np.cumsum(p) / p.sum()
    return np.interp(rng.random(n), cdf, grid)


def test_wham_recovers_known_pmf():
    rng = np.random.default_rng(0)
    grid = np.linspace(4.0, 12.0, 4001)
    true = 3.0 * np.exp(-((grid - 7.0) ** 2) / 0.8) - 2.0 * np.exp(-((grid - 9.5) ** 2) / 0.5)
    r0 = np.arange(4.5, 11.6, 0.5)
    samples = [_sample_biased(true, grid, c, 10.0, 20000, rng) for c in r0]
    res = wham(samples, r0, 10.0, bins=np.linspace(4.5, 11.5, 71), correlated=False)
    ref = np.interp(res.centers, grid, true)
    ok = np.isfinite(res.pmf)
    diff = (res.pmf - ref)[ok]
    assert np.sqrt(np.mean((diff - diff.mean()) ** 2)) < 0.1


# --------------------------------------------------------------- theory -------
def test_bjerrum_and_debye_lengths():
    assert theory.bjerrum_length_nm() == pytest.approx(0.715, abs=0.002)
    assert theory.debye_length_nm(0.1) == pytest.approx(0.961, abs=0.01)
    assert theory.ionic_strength_from_debye_M(theory.debye_length_nm(0.05)) == pytest.approx(0.05)


def test_dlvo_factor_limits():
    assert theory.dlvo_charge_factor(1e-9, 3.0) == pytest.approx(1.0)
    r = np.array([10.0])
    u = theory.yukawa_dlvo(r, 1, 1, 0.0, 0.0, debye=1e9)
    assert u[0] == pytest.approx(0.1)        # bare Coulomb l_B / r
