#!/bin/bash
# Regenerate every figure in figures/ (quick preset; production windows for the PMF).
# Usage: bash scripts/reproduce_figures.sh [N_PARALLEL_RUNS] [RUNS_DIR]
# Cost on an 8-core laptop: about 2–3 hours, dominated by the DPD interface runs.
set -euo pipefail
J=${1:-8}
RUNS=${2:-runs}
S=simulations
mkdir -p figures

# Module 0 — validation
flocmd sweep $S/00_validation/sweep_chain_scaling.toml -j "$J" -o "$RUNS/validation_chain" --analyze
flocmd plot chain_scaling "$RUNS/validation_chain"
flocmd sweep $S/00_validation/sweep_eos.toml -j 3 -o "$RUNS/validation_eos" --analyze
flocmd plot eos "$RUNS/validation_eos"

# Module 1 — dose response, kinetics, movie of the optimal dose
flocmd sweep $S/01_flocculation_kinetics/sweep_dose.toml -j "$J" -o "$RUNS/flocculation_dose" --analyze
flocmd plot flocculation "$RUNS/flocculation_dose" --charge-fraction 1
flocmd movie "$RUNS/flocculation_dose/charge_fraction-1_dose-1_seed-1"

# Module 2 — bridging PMF (production windows; the inward run starts at 6.5 + 25 x 0.4 = 16.5)
sed 's/colloid_separation = 16.0.*/colloid_separation = 16.5/' $S/02_bridging_pmf/sweep_pmf.toml > "$RUNS/sweep_pmf_production.toml"
flocmd sweep "$RUNS/sweep_pmf_production.toml" --preset production -j 4 -o "$RUNS/pmf_production" --analyze
flocmd plot pmf "$RUNS/pmf_production"

# Module 3 — oil/water interface + migration movie
flocmd sweep $S/03_oil_water_interface/sweep_architecture.toml -j "$J" -o "$RUNS/interface_architecture" --analyze
flocmd plot interface "$RUNS/interface_architecture"
flocmd run "$RUNS/movie_interface" --workflow interface --preset quick \
    --set architecture='"random"' --set hydrophobic_fraction=0.2 --set polymer_start='"water"' \
    --set box='[10, 10, 24]' --set n_chains=8 --set n_eq=0 --set n_run=40000 --set movie=1 --set n_movie=500
flocmd movie "$RUNS/movie_interface"

cp "$RUNS/validation_chain/chain_scaling.png" "$RUNS/validation_eos/eos.png" figures/
cp "$RUNS/flocculation_dose/dose_response.png" "$RUNS/flocculation_dose/kinetics.png" figures/
cp "$RUNS/flocculation_dose/charge_fraction-1_dose-1_seed-1/flocculation.gif" figures/
cp "$RUNS/pmf_production/pmf.png" "$RUNS/interface_architecture/interface_summary.png" figures/
cp "$RUNS/interface_architecture/HMPAM_architecture-random_hydrophobic_fraction-0.2_seed-1/profiles.png" \
   figures/interface_profiles.png
cp "$RUNS/movie_interface/interface.gif" figures/
echo "figures written to figures/"
