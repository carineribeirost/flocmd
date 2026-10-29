# Models, methods and validation

This document describes every physical model in `flocmd`, how reduced units
map to real ones, what each observable means, how the codes were validated,
and the known limitations. Equations use $k_BT = 1$.

---

## 1. Coarse-grained colloid + polyelectrolyte model (Modules 1–2)

### 1.1 Particles and units

| Symbol | Meaning | Value (default) | Physical estimate |
|---|---|---|---|
| $\sigma$ | monomer bead diameter, length unit | 1 | ≈ 0.71 nm (≈ 2–3 acrylamide units) |
| $l_B$ | Bjerrum length | 1 $\sigma$ | 0.715 nm, water at 25 °C |
| $d_c$ | colloid diameter | 6 $\sigma$ | ≈ 4.3 nm (silica / clay / oil-wet fines) |
| $Z$ | bare colloid charge | 20 e | ≈ 0.34 e nm⁻², ≈ 0.055 C m⁻² (silica near pH 7–9) |
| $\kappa^{-1}$ | Debye length | 2 $\sigma$ | 1.4 nm ≈ 46 mM 1:1 salt |
| $N$ | chain length | 60 beads | ≈ 40 nm contour |
| $f$ | fraction of cationic beads | 0.25–1 | CPAM “charge density” 25–100 % |
| $\Phi$ | dose: polymer charge / colloid charge | 0–3 | |

`flocmd info` prints the Debye-length ↔ ionic-strength table:
$\kappa^2 = 8\pi l_B N_A I$, so $\kappa^{-1} = 0.304/\sqrt{I[\mathrm{M}]}$ nm.

### 1.2 Interactions

* **Excluded volume.** All pairs interact through the purely repulsive WCA
  potential, written with LAMMPS `lj/expand` so that the colloid core has
  diameter $\approx d_c$:
  $U(r) = 4\varepsilon[(\sigma/(r-\Delta))^{12} - (\sigma/(r-\Delta))^{6}] + \varepsilon$
  for $r - \Delta < 2^{1/6}\sigma$, with $\Delta = d_c-1$ (colloid–colloid) and
  $\Delta = (d_c-1)/2$ (colloid–monomer).
* **Screened electrostatics.** Salt is implicit (Debye–Hückel):
  $U_{ij}(r) = l_B\, q_i q_j\, e^{-\kappa r}/r$. Each charge carries the DLVO
  finite-size factor $e^{\kappa a}/(1+\kappa a)$, so for two spheres of radii
  $a_1, a_2$ the interaction is the linearised Poisson–Boltzmann result
  $l_B Z_1Z_2 \frac{e^{\kappa a_1}}{1+\kappa a_1}\frac{e^{\kappa a_2}}{1+\kappa a_2}\frac{e^{-\kappa r}}{r}$
  (Verwey & Overbeek). Cut-offs are $5\kappa^{-1}$ beyond contact
  ($e^{-5} < 1\%$). The factor is applied in the input (`set type ... charge`),
  so a single data file serves every salt concentration.
* **Optional specific adsorption.** `eps_ads > 0` turns the neutral-monomer /
  colloid pair into a full LJ well of depth $\varepsilon_{ads}$ (e.g. amide
  H-bonding of polyacrylamide to silanol groups). Charged monomers keep the
  WCA core, so electrostatic and non-electrostatic adsorption can be separated.
* **Bonds.** Kremer–Grest FENE, $K = 30$, $R_0 = 1.5$ (no bond crossing).

### 1.3 Dynamics

Langevin (Brownian-like) dynamics, implicit solvent, $\Delta t = 0.01\tau$.
Monomer friction $\gamma_m = 1\, m/\tau$. Colloid friction follows Stokes,
$\gamma_c = d_c \gamma_m$ (LAMMPS `fix langevin ... scale 1 dc²`, because the
colloid mass is $d_c^3$). The colloid diffusion coefficient is therefore
$D_c = 1/d_c$, used for the Smoluchowski reference rate.

No hydrodynamic interactions are included (see Limitations).

### 1.4 Protocol (Module 1)

A jar-test analogue:

1. **Soft push-off** of bead overlaps (ramped cosine potential, colloids fixed).
2. **Rapid mix.** Chains adsorb onto the still-dispersed colloids (colloids fixed).
3. **Slow mix.** Everything moves; colloids collide, bridge and flocculate.
   $t = 0$ of the kinetics is the start of this stage.

### 1.5 Observables (Module 1)

| Quantity | Definition |
|---|---|
| contact cluster | colloids with surface gap < 2 $\sigma$ (also tracked live in LAMMPS with `compute cluster/atom`) |
| floc | connected component of the graph of contacts **or** shared bridging chains |
| $\bar s_w$ | weight-average floc size $\sum s^2/\sum s$ |
| adsorbed monomer | centre within $d_c/2 + 1.5\sigma$ of a colloid centre |
| bridging chain | chain adsorbed on ≥ 2 different colloids |
| trains / loops / tails | Scheutjens–Fleer decomposition of each adsorbed chain |
| floc net charge | colloid charges + charges of all chains adsorbed in the floc, per colloid; $>0$ = **charge reversal** |
| $\tau_{agg}$, $k$ | fit of $N(t) = N_0/(1+t/\tau)$, $k = 2/(n_0\tau)$ (constant-kernel Smoluchowski) |
| stability ratio $W$ | $k_{fast}/k$ with $k_{fast} = 16\pi D_c a$ (diffusion-limited, identical spheres), $a = (d_c + 2\sigma)/2$: the same capture distance at which a contact cluster is counted, so barrier-free aggregation gives $W = 1$ |

The physical questions are: (i) where the optimal dose is and how it depends
on charge density and salt; (ii) whether flocs form by **charge-patch**
neutralisation (flat trains, few bridges, net floc charge ≈ 0) or by
**bridging** (loops and tails spanning the gap, many bridging chains);
(iii) whether overdosing restabilises the dispersion by charge reversal
(Gregory 1973; Bolto & Gregory 2007).

### 1.6 Bridging free energy (Module 2)

Two colloids, a fixed number of chains, and an umbrella bias
$U_w = \tfrac{K}{2}(r - r_{0,w})^2$ on the centre-of-mass distance
(`fix spring couple`, $K = 10\,k_BT/\sigma^2$, $\Delta r_0 = 0.5\sigma$).
Windows are visited sequentially: the run is a slow pull followed by
equilibrium sampling in each window. Running the windows outward
(`direction = +1`) and inward (`-1`) tests for hysteresis, which is expected
when bridges are long-lived.

The unbiased distribution is obtained with WHAM (Kumar et al. 1992). The
statistical inefficiency $g = 2\tau_{int}$ of every window is estimated, and
errors come from a moving-block bootstrap. Because $r$ is a 3-D distance,
the pair PMF is $w(r) = -k_BT\ln P(r) + 2k_BT\ln r$.

**Validation.** Without polymer the PMF must equal the bare pair potential
(WCA + DLVO Yukawa), which is known analytically (`theory.colloid_pair_potential`).
`flocmd analyze pmf` reports the RMSD between the two.

---

## 2. DPD oil/water/flocculant model (Module 3)

### 2.1 Model

Groot–Warren DPD: soft conservative force $F^C = a_{ij}(1 - r/r_c)\hat{\mathbf r}$,
pairwise dissipative and random forces ($\gamma = 4.5$, $k_BT = 1$), density
$\rho = 3$, $\Delta t = 0.04$. With 3 water molecules per bead,
$r_c = 3.107(\rho N_m)^{1/3}$ Å ≈ 0.65 nm (Groot & Rabone 2001).

Repulsions follow $a_{ij} = 25 + 3.27\chi_{ij}$ (Groot & Madden 1998). Defaults:

|   | W | O | P | H |
|---|---|---|---|---|
| **W** water | 25 | 80 | 26 | 80 |
| **O** oil (trimer ≈ decane) | | 25 | 60 | 25 |
| **P** acrylamide backbone | | | 25 | 45 |
| **H** alkyl sticker | | | | 25 |

$\chi$ can be estimated from solubility parameters
(`theory.chi_from_solubility`) or, better, from atomistic simulations
(see `roadmap.md`). Bonds are harmonic, $E = K (r - r_0)^2$ with $K = 25$,
$r_0 = 0.7\,r_c$, giving bond lengths of $0.76 \pm 0.1\,r_c$. Bonded beads keep
their pair forces (`special_bonds lj 1 1 1`).

> **Pitfall: soft bonds and ghost atoms.** The original Groot–Warren spring
> ($E = 2r^2$, i.e. $C = 4$) lets bonds fluctuate up to ~3 $r_c$, beyond the
> default ghost-atom cutoff ($r_c$ + skin = 1.3). On a single MPI rank LAMMPS
> does not stop with "bond atoms missing": it uses the only copy of the partner
> it has, on the far side of the periodic box. The bond then pulls atoms through
> the boundary and corrupts image flags (unwrapped bonds of 10–20 $r_c$), which
> breaks every conformational analysis. The stiffer bond and
> `comm_modify cutoff 2.5` remove the problem; `tests` and the analysis check
> that unwrapped bond lengths stay below 1.5 $r_c$.

### 2.2 Observables

* **Interfacial tension** (mechanical route, two interfaces):
  $\gamma = \frac{L_z}{2}\left[P_{zz} - \tfrac12(P_{xx}+P_{yy})\right]$,
  block-averaged (Flyvbjerg & Petersen 1989) for an honest error bar.
* **Density profiles** of each bead type; the oil profile is fitted with a
  two-interface tanh to locate the Gibbs dividing surfaces and the width.
* **Adsorption**: a chain is adsorbed when any bead is within $1 r_c$ of a
  Gibbs surface (or inside the oil).
* **Conformation**: $R_{g,\parallel}^2 = S_{xx}+S_{yy}$, $R_{g,\perp}^2 = S_{zz}$,
  anisotropy $\sqrt{R_{g,\parallel}^2/2R_{g,\perp}^2}$ (1 for an isotropic coil).
* **Sticker penetration**: mean signed distance of H beads to the interface
  (negative = inside the oil).

The oil slab is kept centred (`fix recenter`) so profiles are not smeared by
slab diffusion.

---

## 3. Validation (Module 0)

| Test | Expectation | Where |
|---|---|---|
| Single Kremer–Grest chain, good solvent | $R_g \propto N^{\nu}$, $\nu \approx 0.588$ | `in.chain_scaling.lmp` |
| DPD equation of state | $(p-\rho k_BT)/(a\rho^2)$ independent of $a$ | `in.dpd_eos.lmp` |
| Bare colloid PMF | WHAM = analytic WCA + DLVO | Module 2, case `bare` |
| WHAM | recovers a known double-well PMF from exact biased samples | `tests/test_analysis.py` |
| Block averaging | AR(1) SEM within 25 % of the exact value | `tests/test_analysis.py` |
| Cluster analysis | periodic images handled | `tests/test_analysis.py` |

**A note on the DPD equation of state.** Groot & Warren fitted
$p = \rho k_BT + \alpha a\rho^2$ with $\alpha = 0.101$. We find that
$\alpha$ is independent of $a$ to within 2 % (a = 15–60), as the scaling
predicts, but that it depends on density: $\alpha$ = 0.092 ($\rho$ = 3),
0.096 ($\rho$ = 4) and 0.098 ($\rho$ = 5), approaching 0.101 only at high
density (at $\rho = 3$, $a = 25$: $p = 23.8$ instead of 25.7). This is not an
integration artefact: $\alpha$ does not change between $\Delta t = 0.04$ and
$0.01$, and it is identical when the virial is computed from the conservative
forces alone, from dumped configurations. The kinetic temperature is
$1.03\,k_BT$ at $\Delta t = 0.04$ (standard velocity Verlet), and this is
accounted for in $\alpha$. The test checks the scaling and reports the
measured $\alpha$ rather than forcing agreement with the fit.

**Stability ratio.** $W$ comes from a constant-kernel Smoluchowski fit to
the early-time decay of the contact-cluster count. When aggregation is
reversible and the count reaches a plateau (under- and overdosed systems),
$W$ is an *effective* early-time value.

---

## 4. Limitations

* **No hydrodynamics** in Modules 1–2 (Langevin). Absolute aggregation rates
  are therefore approximate; *relative* trends (dose, charge density, salt)
  are the robust output. DPD or SRD solvents (`roadmap.md`) would add
  hydrodynamic interactions.
* **Mean-field salt** (Debye–Hückel): no ion correlations, no counterion
  condensation, no specific ion effects (e.g. Ca²⁺ bridging, important in
  produced water). Explicit ions with PPPM are a natural extension.
* **Size gap**: real flocculated particles are 0.1–100 µm; here colloids are
  nanometric. Results should be read as mechanisms and scaling trends, not as
  quantitative jar-test predictions.
* **DPD parameters** are generic (literature-typical $\chi$ values), not yet
  fitted to a specific crude oil or polymer.

---

## References

* Thompson, A. P. et al. LAMMPS – a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. *Comput. Phys. Commun.* **271**, 108171 (2022).
* Kremer, K.; Grest, G. S. Dynamics of entangled linear polymer melts: a molecular-dynamics simulation. *J. Chem. Phys.* **92**, 5057 (1990).
* Grest, G. S.; Kremer, K. Molecular dynamics simulation for polymers in the presence of a heat bath. *Phys. Rev. A* **33**, 3628 (1986).
* Groot, R. D.; Warren, P. B. Dissipative particle dynamics: bridging the gap between atomistic and mesoscopic simulation. *J. Chem. Phys.* **107**, 4423 (1997).
* Groot, R. D.; Madden, T. J. Dynamic simulation of diblock copolymer microphase separation. *J. Chem. Phys.* **108**, 8713 (1998).
* Groot, R. D.; Rabone, K. L. Mesoscopic simulation of cell membrane damage, morphology change and rupture by nonionic surfactants. *Biophys. J.* **81**, 725 (2001).
* Verwey, E. J. W.; Overbeek, J. Th. G. *Theory of the Stability of Lyophobic Colloids*. Elsevier (1948).
* von Smoluchowski, M. Versuch einer mathematischen Theorie der Koagulationskinetik kolloider Lösungen. *Z. Phys. Chem.* **92**, 129 (1917).
* Gregory, J. Rates of flocculation of latex particles by cationic polymers. *J. Colloid Interface Sci.* **42**, 448 (1973).
* Bolto, B.; Gregory, J. Organic polyelectrolytes in water treatment. *Water Res.* **41**, 2301 (2007).
* Scheutjens, J. M. H. M.; Fleer, G. J. Statistical theory of the adsorption of interacting chain molecules. 1. *J. Phys. Chem.* **83**, 1619 (1979).
* Fleer, G. J.; Cohen Stuart, M. A.; Scheutjens, J. M. H. M.; Cosgrove, T.; Vincent, B. *Polymers at Interfaces*. Chapman & Hall (1993).
* Dobrynin, A. V.; Rubinstein, M. Theory of polyelectrolytes in solutions and at surfaces. *Prog. Polym. Sci.* **30**, 1049 (2005).
* Torrie, G. M.; Valleau, J. P. Nonphysical sampling distributions in Monte Carlo free-energy estimation: umbrella sampling. *J. Comput. Phys.* **23**, 187 (1977).
* Kumar, S. et al. The weighted histogram analysis method for free-energy calculations on biomolecules. I. The method. *J. Comput. Chem.* **13**, 1011 (1992).
* Flyvbjerg, H.; Petersen, H. G. Error estimates on averages of correlated data. *J. Chem. Phys.* **91**, 461 (1989).
* Kirkwood, J. G.; Buff, F. P. The statistical mechanical theory of surface tension. *J. Chem. Phys.* **17**, 338 (1949).
