# Installing and reproducing

## 1. Install

`flocmd` needs Python ≥ 3.10 and a LAMMPS executable that includes the
MOLECULE and DPD-BASIC packages (every standard build does).

### Linux, WSL2 or macOS: pip (simplest)

```bash
python -m venv ~/.venvs/flocmd
source ~/.venvs/flocmd/bin/activate
pip install -e ".[dev,lammps]" mpich     # LAMMPS 22Jul2025 wheel + the MPI runtime it links to
echo "info config" | lmp -log none | head  # check
```

The `lammps` wheel installs an `lmp` executable; `mpich` provides `libmpi.so.12`.

If `lmp` aborts at start-up with `MPI_Init ... failed` and `UCX ERROR`
messages, the MPICH runtime is trying to use a network device it cannot open
(common on cloud VMs with RDMA adapters). Restrict it to shared memory and TCP:

```bash
export UCX_TLS=self,sm,tcp
```

### conda-forge

```bash
conda env create -f environment.yml
conda activate flocmd
pip install -e .
```

### Windows

Either use WSL2 (recommended: identical to a Linux cluster), or install the
official LAMMPS Windows package (lammps.org → Download → Windows) and point
`flocmd` to it:

```powershell
$env:FLOCMD_LMP = "C:\Program Files\LAMMPS 64-bit 22Jul2025\bin\lmp.exe"
```

### Choosing how LAMMPS is launched

`flocmd` uses `$FLOCMD_LMP` if set, otherwise the first of `lmp`,
`lmp_serial`, `lmp_mpi`, `lmp.exe` on `PATH`. Examples:

```bash
export FLOCMD_LMP="mpirun -np 4 lmp"            # MPI
export FLOCMD_LMP="lmp -sf omp -pk omp 4"       # OpenMP
```

`flocmd info` shows what will be used and prints the unit mapping.

## 2. Test

```bash
pytest -v        # unit tests + end-to-end LAMMPS smoke tests (~15 s)
```

## 3. Run the studies

Every study is a TOML sweep. `-j` sets how many LAMMPS runs execute
concurrently (one core each). Presets: `smoke` (seconds), `quick` (laptop:
minutes per run), `production` (cluster: hours per run).

```bash
# Module 0 — validation
flocmd sweep simulations/00_validation/sweep_chain_scaling.toml -j 8 --analyze
flocmd plot chain_scaling runs/validation_chain
flocmd sweep simulations/00_validation/sweep_eos.toml -j 3 --analyze
flocmd plot eos runs/validation_eos

# Module 1 — flocculation kinetics: dose and salt
flocmd sweep simulations/01_flocculation_kinetics/sweep_dose.toml -j 8 --analyze
flocmd plot flocculation runs/flocculation_dose --charge-fraction 1
flocmd movie runs/flocculation_dose/charge_fraction-1_dose-1_seed-1

# Module 2 — bridging PMF (umbrella sampling + WHAM)
flocmd sweep simulations/02_bridging_pmf/sweep_pmf.toml -j 4 --analyze
flocmd plot pmf runs/pmf

# Module 3 — oil/water interface (DPD)
flocmd sweep simulations/03_oil_water_interface/sweep_architecture.toml -j 4 --analyze
flocmd plot interface runs/interface_architecture
```

To run a single configuration by hand:

```bash
flocmd build flocculation -o runs/test --preset quick --set dose=1.5 --set charge_fraction=0.5
cd runs/test && lmp -in ../../simulations/01_flocculation_kinetics/in.flocculation.lmp \
      -var common ../../simulations/common -var data data.lmp -var n_run 200000
flocmd analyze runs/test
```

(`lammps_vars.json` in the run directory lists every `-var` the workflow would pass.)

## 4. On a cluster (SLURM)

```bash
flocmd sweep simulations/01_flocculation_kinetics/sweep_dose.toml \
       --preset production -o runs/dose_prod --dry-run          # only prepares run dirs + jobs.txt
sbatch --array=1-$(wc -l < runs/dose_prod/jobs.txt)%50 scripts/slurm_array.sh runs/dose_prod/jobs.txt
flocmd analyze runs/dose_prod && flocmd plot flocculation runs/dose_prod
```

## 5. Output files

| Module | Raw LAMMPS output | Written by `flocmd analyze` |
|---|---|---|
| 01 | `clusters.dat`, `rdf_cc.dat`, `traj.lammpstrj` | `timeseries.csv`, `summary.json` |
| 02 | `windows.txt`, `window_<w>.dat` | `pmf.csv`, `summary.json` |
| 03 | `ift.dat`, `prof_*.dat`, `polymer.lammpstrj`, `movie.lammpstrj` | `profiles.csv`, `summary.json` |
| 00 | `eos.dat`, `rg.dat` | `eos.csv`, `summary.json` |

Every sweep directory also gets `summary.csv` (one row per run) for your own
analysis in pandas, R or Excel. Trajectories open directly in OVITO or VMD.
