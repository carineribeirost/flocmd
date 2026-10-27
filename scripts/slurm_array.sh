#!/bin/bash
# -----------------------------------------------------------------------------
# Run a prepared flocmd sweep as a SLURM job array (one array task per run dir).
#
#   flocmd sweep simulations/01_flocculation_kinetics/sweep_dose.toml \
#          --preset production -o runs/dose_prod --dry-run
#   sbatch --array=1-$(wc -l < runs/dose_prod/jobs.txt)%50 scripts/slurm_array.sh runs/dose_prod/jobs.txt
#   # afterwards:
#   flocmd analyze runs/dose_prod && flocmd plot flocculation runs/dose_prod
#
# Adapt partition/account/modules to your cluster.
# -----------------------------------------------------------------------------
#SBATCH --job-name=flocmd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail
JOBS_FILE=${1:?usage: sbatch --array=1-N slurm_array.sh jobs.txt}
RUN_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")

# module load lammps            # or: source ~/venvs/flocmd/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# OpenMP-accelerated styles (OPENMP package). For MPI instead use: FLOCMD_LMP="srun lmp"
export FLOCMD_LMP="lmp -sf omp -pk omp ${SLURM_CPUS_PER_TASK}"

echo "[$(date)] running ${RUN_DIR}"
flocmd run "${RUN_DIR}"
flocmd analyze "${RUN_DIR}" > "${RUN_DIR}/analysis.log"
echo "[$(date)] done"
