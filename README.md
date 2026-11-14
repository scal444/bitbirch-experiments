# BitBIRCH implementations: speed and clustering quality

Scripts to run four BitBIRCH implementations on the same fingerprint libraries and compare how fast they are and
how good their clusters are.

| implementation | what it is |
|---|---|
| **bblean serial** | the reference: [bblean](https://github.com/mqcomplab/bblean), one molecule at a time (C++ tree) |
| **bblean split-merge** | bblean's own parallel mode (`bb multiround`): cluster parts of the input separately, then merge |
| **bblean batched** | the batched algorithm in bblean as a multithreaded CPU fit; same clusters as nvMolKit batched |
| **nvMolKit batched** | [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit)'s batched BitBIRCH on one GPU |

## How quality is judged

The main measure is quality against the chemistry: within-cluster similarity (iSIM), scaffold purity, and AMI/ARI
of the clusters against Murcko scaffolds and ChEMBL targets.

The second is agreement between runs. BitBIRCH depends on input order, so bblean serial isn't ground truth. Like the
BitBIRCH authors, the scripts measure how well two runs' clusters agree (AMI and ARI; 1 = same clusters, 0 =
chance). The floor is bblean serial against itself on a different input order: that's as close as a new
implementation can be expected to get. Every compared pair (serial vs serial, X vs X, serial vs X) uses two
different input orders, so each includes the same order change as the floor. Scores cover all molecules, and also
just the molecules in the 20 and 50 largest clusters.

## How to reproduce

1. **Setup.** Follow [`setup/SETUP.md`](setup/SETUP.md): `setup/install_env.sh`, then `setup/fetch_and_build.sh all`.
   Data downloads and prep are in [`docs/DATA.md`](docs/DATA.md).
2. **Config.** `cp config/paths.example.env config/paths.env` and fill in `BB_DATA`, `BB_OUT` (labels, outside the
   repo), the build paths, and `BB_RESULTS_NAME`. Results go to `results_<BB_RESULTS_NAME>/`, laid out like the
   example results folder.
3. **Run.** Set the machine variables at the top of each script (threads, processes), then:
   ```
   bash scripts/run/run_quality_all.sh   # labels, quality metrics, MoleculeACE, refinement
   bash scripts/run/run_timings.sh       # time and peak memory; use an otherwise idle machine
   ```
   Each takes many hours at full size. Both take stage names to run only part. The quality script skips outputs
   that already exist; the timing script appends a row per run. Stages and single-run commands:
   [`scripts/README.md`](scripts/README.md).
4. **Plot.** `python scripts/plot/make_all.py` writes all figures (PNG) and `plots/INDEX.md`. It only reads the
   tables under `data/`, so it can redraw any results folder's figures without the fingerprints or a GPU.

Clusters don't depend on the hardware (same input order and threshold give the same clusters); times do. If you
publish timings, state the system.

## Example results

Data and plots for 1× NVIDIA GB300 GPU (251 GiB) + 72-core NVIDIA Grace CPU (Arm Neoverse-V2, 486 GiB), driver
615.71.09, CUDA 13.3 are in [`results_gb300_72coreARM/`](results_gb300_72coreARM/); see its
[README](results_gb300_72coreARM/README.md).

## Layout

| folder | contents |
|---|---|
| `results_<name>/` | one results folder: `data/` (tables) and `plots/` (figures) |
| `scripts/` | data prep, runs, metrics, refinement, figures |
| `setup/` | environment and build scripts |
| `config/` | paths template |
| `docs/DATA.md` | where the libraries come from and how to prepare them |

Fingerprints, labels and timing inputs aren't in git; they live under the paths in `config/paths.env`.
