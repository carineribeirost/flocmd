# Roadmap — from mechanisms to a predictive multiscale model

The current modules isolate mechanisms with generic parameters. The steps
below would turn them into a chemistry-specific, predictive workflow for
produced-water treatment. Each is scoped as a self-contained work package.

## WP1 — Atomistic parameterisation (bottom-up)

* All-atom MD of PAM, cationic PAM (AETAC/DMC co-monomers), anionic HPAM and
  HM-PAM oligomers in TIP4P/2005 water and at the n-decane/water interface
  (OPLS-AA or GAFF2; topologies via LigParGen / Moltemplate / fftool + Packmol;
  PPPM electrostatics).
* Outputs: oligomer–oil and oligomer–water solvation free energies, interfacial
  tension reduction per adsorbed segment, monomer–silica adsorption free energy
  (for `eps_ads`).
* Map them onto the coarse models: DPD $\chi_{ij}$ from mixing free energies,
  `eps_ads` from adsorption PMFs, chain stiffness from atomistic persistence length.

## WP2 — Explicit ions and produced-water brines

* Replace Debye–Hückel with explicit Na⁺/Cl⁻/Ca²⁺ and PPPM in Module 1.
* Questions: Ca²⁺-mediated bridging of anionic polymers onto negative
  particles; counterion condensation on high-charge CPAM; the salt
  concentration above which the optimal dose disappears.

## WP3 — Hydrodynamics and larger flocs

* Explicit DPD (or SRD, `fix srd`) solvent around rigid colloids (`fix rigid`)
  to obtain hydrodynamically correct aggregation rates and floc fractal
  dimensions; shear flow (`fix deform` + SLLOD) to study floc breakage and
  regrowth, the real operating condition of flotation units and hydrocyclones.

## WP4 — Charged DPD and oil droplets

* **Hypothesis from Module 3:** distributed stickers flatten HM-PAM on the
  interface (lower γ), whereas end-anchored architectures (telechelic,
  diblock) leave hydrophilic blocks dangling into the water, which are
  available to bridge a second droplet. Test it with droplet–droplet PMFs
  (umbrella sampling on two rigid or soft oil droplets, as in Module 2) for each
  architecture, at matched adsorbed amount.
* Smeared-charge DPD (`pair_style coul/slater/long` + `kspace_style pppm`) for
  cationic flocculants at negatively charged oil droplets.
* Multi-droplet emulsions with natural surfactants (asphaltene/resin-like
  amphiphiles) to quantify demulsification and oil-droplet flocculation rates.

## WP5 — Data-driven formulation

* Use the sweep engine to build a dataset (architecture × dose × salinity →
  floc size, PMF depth, IFT) and train a surrogate model for formulation
  screening, with active learning to pick the next simulations.
* Compare against jar tests, zeta potential and turbidity data.
