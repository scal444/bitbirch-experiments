# scripts/

Paths come from `config/paths.env` (read by `common/paths.py`). Tables go to `results_<BB_RESULTS_NAME>/data/`,
figures to its `plots/`. Run everything in the `bitbirch` conda environment from `setup/`, with the builds from
`setup/fetch_and_build.sh` (bblean at pinned commits, upstream nvMolKit main).

The four implementations have one name each, used in every table, figure script and command line
(`common/implementations.py`). Split-merge's process count P is a separate `processes` column. The label folders
under `BB_OUT/labels/<library>/` keep shorter internal names:

| implementation | name | label folder |
|---|---|---|
| bblean serial | `bblean_serial` | `bblean_tree` |
| bblean split-merge | `bblean_split_merge` | `bblean_mr<P>` |
| bblean batched | `bblean_batched` | `bblean_batched` |
| nvMolKit batched | `nvmolkit_batched` | `nvmolkit_main` |

## Running

Order: prepare data (`data/`, see `docs/DATA.md`), then `run/run_quality_all.sh`, `run/run_timings.sh`, and
`plot/make_all.py`. Thresholds come from `data/thresholds/isim_sigma.json` in the results folder (written by
`metrics/thresholds.py`); a results folder without one uses the example results folder's.

`bash scripts/run/run_quality_all.sh [stages]`: set `THREADS` (bblean batched) and `P_MAIN` (split-merge processes)
at the top. Labels go to `BB_OUT/labels/`.

| stage | writes |
|---|---|
| labels | labels of all four; `data/checks/bblean_batched_vs_nvmolkit.csv` (the two batched ones must match) |
| metrics | `data/quality/by_run/`, `quality.csv`, `top_clusters_by_run.csv`, `top_clusters.csv`, `size_bins.csv`, `agreement.csv` |
| moleculeace | `data/moleculeace/` (needs the MoleculeACE sets) |
| refine | `data/refinement/` (labels and working figures under `BB_REFINE_POST_LABELS`) |

`bash scripts/run/run_timings.sh [stages]` (stages: `inputs gpu sweep cpu big finalize`): set `THREADS`, `P_MAIN`,
`WORKERS` at the top, and drop sizes from `stage_big` that won't fit in your memory. Run it on an otherwise idle
machine. Rows are appended to
`data/timing/timings.csv`; `finalize` writes `scaling_exponents.csv`.

Single runs, e.g. ChEMBL34 at k = 3.5, input order 0:

```
python scripts/run/run_bblean.py bblean_serial chembl34 3.5 --seeds 0
python scripts/run/run_bblean.py bblean_batched chembl34 3.5 --seeds 0 --threads 36
python scripts/run/run_nvmolkit.py chembl34 3.5 --seeds 0
python scripts/run/run_bblean_multiround.py chembl34 3.5 --seeds 0 --files 36 --ps 36
python scripts/metrics/metrics.py chembl34    # skips labellings that already have a row; --force redoes them
python scripts/run/perf_run.py nvmolkit_batched chembl34 --repeats 2   # one timed run (after perf_inputs.py)
```

## What's where

- `common/`: `paths.py` (config, results folder, libraries, thresholds, input orders, label layout),
  `implementations.py` (the four names and their label folders), `partitions.py` (compare two labellings).
- `data/`: fingerprint and scaffold prep (`docs/DATA.md`); `zinc22_export.py` copies the small ZINC22 run summaries
  into `data/zinc22/`.
- `run/`: the two drivers above; `run_bblean.py`, `run_bblean_multiround.py`, `run_nvmolkit.py` (labels);
  `run_moleculeace.py`; `check_batched_vs_nvmolkit.py` (bblean batched and nvMolKit batched must give the same
  clusters); timing helpers `perf.py` (timing sets), `perf_inputs.py`,
  `perf_run.py` (one timed run), `perf_finalize.py`.
- `metrics/`: `thresholds.py`; `metrics.py` (quality against chemistry, the main quality measure, GPU);
  `agreement.py` (agreement between runs, GPU); `ext_metrics.py` (torch ARI/AMI, matches scikit-learn);
  `validate_metrics.py`; `combine_quality.py`; `top_clusters.py`; `paper_indices.py` and `moleculeace_metrics.py`
  (BitBIRCH-paper indices).
- `refinement/`: bblean's refinement on each implementation's tree: `run_refine.py`, `analyze.py`, `summarize.py`,
  `plot.py` (working figures, outside the repo).
- `plot/`: `make_all.py` (combined tables, every figure as PNG, `plots/INDEX.md`); `common.py` (colours, loaders, and
  `FIGURES`, which puts each figure in `plots/<topic>/<level>/`); `quality/`, `timing/`, `other/`.
