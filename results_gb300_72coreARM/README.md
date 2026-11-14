# Results: one GB300 workstation (72-core Arm CPU)

Made with the scripts in this repo. `data/` holds the tables, `plots/` the figures as PNG
([`plots/INDEX.md`](plots/INDEX.md) lists them all; each topic has `primary/`, `secondary/`, `extra/`, the main ones
in `plots/*/primary/`).

**System:** Dell Pro Max with GB300. 1× NVIDIA GB300 GPU (251 GiB), NVIDIA Grace CPU with 72 Arm Neoverse-V2 cores
(up to 3.3 GHz, compute capability 10.3), 486 GiB host memory, driver 615.71.09, CUDA 13.3, GCC 13.3, Ubuntu 24.04.
bblean builds at the commits in `setup/SETUP.md`, nvMolKit at commit 763b923 of its `main` branch (the setup now
builds the latest `main`), default compiler flags. Python 3.13.15, torch 2.14.1+cu130, RDKit
2025.09.6, NumPy 2.2.6, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, matplotlib 3.11.2, bblean 0.10.3 (for
thresholds and as a reference). Times depend on this hardware; clusters don't, but the thread and process sweeps
(up to 72) depend on the core count.

**Settings:** t = iSIM + k·σ per library (`data/thresholds/isim_sigma.json`), k = 3.5 unless stated (quality also
uses 3 and 4). Branching factor 1024, diameter merge, batch size 4096 for both batched implementations. Split-merge:
36 parts, 36 processes. bblean batched: 36 threads for the size comparison, 1–72 in the thread sweep. Input order
`s` = `numpy.random.default_rng(s).permutation(N)`. MoleculeACE uses the BitBIRCH paper's settings instead (below).

## Timing

| | 101M ZINC20 molecules | 10M Enamine REAL molecules |
|---|---|---|
| bblean serial (1 thread) | 1,459 s | 198 s |
| bblean split-merge | 157 s (36 processes) | 95 s (best, 16 processes) |
| bblean batched | 57.5 s (36 threads) | 5.0 s (72 threads) |
| nvMolKit batched (1 GPU) | 19.9 s | 3.5 s |

## Findings

- **Speed.** nvMolKit batched is 73× faster than bblean serial at 101M molecules. bblean batched scales with threads
  (29× its one-thread speed on Enamine 10M). Split-merge stops getting faster past 16 processes.
- **Chemistry.** Batched clusters have nearly the same within-cluster similarity, scaffold purity and agreement with
  Murcko scaffolds and ChEMBL targets as bblean serial. Split-merge gives more, looser clusters that match scaffolds
  and targets less well.
- **Agreement with bblean serial.** On all five libraries the batched algorithm agrees with bblean serial at or close
  to the floor. ChEMBL34 at k = 3.5 (10 input orders): AMI 0.275 vs a floor of 0.280 on all molecules, 0.467 vs
  0.506 on the 20 largest clusters. Split-merge is below the floor on every library, clearly on four
  (ChEMBL34: 0.146 and 0.354) and slightly on ChEMBL33 NP.
- **Scale.** nvMolKit batched clustered all 12.16 billion molecules of 2D ZINC22 in 94 minutes on the one GPU. That
  run was done outside these scripts; the repo has its data prep and the export of its tables (`data/zinc22/`).
- **Split-merge.** On the `cpp-tree` build its clusters match bblean 0.10.3's and it's 4.4× faster on ChEMBL34. The
  looser merge test in its later rounds gives a coarser clustering: ChEMBL34 at k = 3.5, input order 0 gives 272,185
  clusters vs 174,948 for serial.

Which nvMolKit build made which rows: the timings and the ChEMBL34 input orders 3–9 (k = 3.5) used commit 763b923;
every other nvMolKit clustering (orders 0–2, or 0–4 on ChEMBL35 additions and ChEMBL33 NP, at k = 3, 3.5 and 4) came
from an earlier development build, whose metrics are in `data/quality/by_run_earlier_labels/`. Both builds give
exactly the same clusters as bblean batched on all 64 of these (`data/checks/bblean_batched_vs_nvmolkit.csv`), so
the build makes no difference to any quality number.

## Agreement between runs

Mean AMI at k = 3.5 (all molecules / 20 largest clusters). Each pair uses two different input orders; mean over all
pairs of orders with labels (ChEMBL34: 10 orders, so 45 pairs within one implementation and 90 against serial).
Pairs sharing an order aren't independent, so the 95 % intervals in the figures are on the narrow side.

| library | serial vs itself (floor) | batched vs itself | bblean serial vs batched | split-merge vs itself | bblean serial vs split-merge |
|---|---|---|---|---|---|
| ChEMBL34 (10 orders) | 0.280 / 0.506 | 0.280 / 0.478 | 0.275 / 0.467 | 0.116 / 0.331 | 0.146 / 0.354 |
| SCUBIDOO (3) | 0.449 / 0.657 | 0.447 / 0.621 | 0.446 / 0.657 | 0.202 / 0.458 | 0.274 / 0.515 |
| Enamine 1M (3) | 0.109 / 0.353 | 0.116 / 0.346 | 0.109 / 0.359 | 0.018 / 0.171 | 0.041 / 0.193 |
| ChEMBL35 additions (5) | 0.453 / 0.677 | 0.450 / 0.673 | 0.450 / 0.688 | 0.319 / 0.607 | 0.362 / 0.662 |
| ChEMBL33 NP (5) | 0.555 / 0.718 | 0.573 / 0.738 | 0.558 / 0.711 | 0.539 / 0.667 | 0.530 / 0.701 |

"Batched" covers bblean batched and nvMolKit batched: identical clusters (`data/checks/`), so identical values (the
figures show both).

Implementation names in every table: `bblean_serial`, `bblean_split_merge` (its process count P in a separate
`processes` column; empty for the others), `bblean_batched`, `nvmolkit_batched`.

Library keys used in file names and tables: `chembl34` (ChEMBL34), `scu` (SCUBIDOO), `ena1m` (Enamine REAL 1M),
`c35` (ChEMBL35 additions), `np` (ChEMBL33 natural products); timing also `ena10m`, `zinc10m`/`25m`/`50m`/`101m`.
Details in `docs/DATA.md`.

## data/quality/

Labels themselves live under `BB_OUT/labels`, not in git.

| file | what | made by |
|---|---|---|
| `agreement.csv` | agreement between two runs on different input orders | `scripts/metrics/agreement.py` |
| `by_run/<ds>.csv`, `by_run/topk_<ds>.csv` | one row per clustering; the same restricted to its 5/20/100/1000 largest clusters | `scripts/metrics/metrics.py` |
| `by_run_earlier_labels/` | same, for labels from an earlier run that give identical partitions (`data/checks/`): nvMolKit batched from the earlier development build, and bblean serial from bblean 0.10.3 (superseded by the `by_run/` rows) | earlier `metrics.py` run |
| `quality.csv`, `top_clusters_by_run.csv` | all `by_run*` files combined (a `by_run/` row wins over an earlier one) | `scripts/metrics/combine_quality.py` (also run by `make_all.py`) |
| `top_clusters.csv`, `size_bins.csv` | the 20 largest clusters of each clustering; cluster counts by size | `scripts/metrics/top_clusters.py` |

**agreement.csv** (one row per pair of runs and subset). Within one implementation: every pair of orders a < b.
Serial vs another implementation: serial on a, the other on b, every a ≠ b.

| column | meaning |
|---|---|
| dataset, k | library key, threshold multiplier |
| implementation_a, implementation_b | `bblean_serial`, `bblean_split_merge` (P = 36), `bblean_batched`, `nvmolkit_batched`; across implementations `implementation_a` is `bblean_serial` |
| order_a, order_b | input orders (always different) |
| subset | `all` (every molecule, singletons included) or `top20` / `top50` (molecules of the 20 / 50 largest clusters of the reference run, both runs restricted to them) |
| ami, ari | adjusted mutual information, adjusted Rand index |
| n_molecules | molecules in the subset |
| clusters_a, clusters_b | clusters of each run among those molecules |

The reference run for `top20`/`top50` is run a (bblean serial). Within one implementation the values are the mean of
both choices of reference. Ties in cluster size go to the lower cluster id.

**quality.csv and by_run/<ds>.csv** (one row per clustering):

| column | meaning |
|---|---|
| dataset | library key |
| implementation | `bblean_serial`, `bblean_split_merge`, `bblean_batched`, `nvmolkit_batched` |
| processes | `bblean_split_merge` only: P (36 everywhere; 10 and 72 too on ChEMBL34 and SCUBIDOO at k = 3.5, order 0, not plotted) |
| bf | branching factor (1024) |
| k, threshold, seed | threshold settings and input order |
| n_molecules | library size |
| seconds | clustering time stored with the labels; not a timing (use `data/timing`) |
| clusters, singletons, singleton_fraction, singletons_per_cluster, largest_cluster, nonsingleton_clusters | cluster counts |
| isim | within-cluster iSIM of clusters with 2+ members, weighted by size |
| library_isim | iSIM of the whole library |
| scaffold_purity | share of molecules in clusters of 2+ that carry the cluster's most common Murcko scaffold |
| medoid_isim_all, medoid_isim_nonsingleton | iSIM of the cluster medoids (lower = clusters further apart) |
| tgt_molecules | molecules with exactly one ChEMBL single-protein target (`chembl34`, `c35` only) |
| scaf_ari, scaf_ami, scaf_v_measure, scaf_homogeneity, scaf_completeness | agreement with Murcko scaffolds |
| tgt_* | the same against ChEMBL targets, on `tgt_molecules` only |

`by_run/` also has bookkeeping columns: `metrics_seconds`, `gpu_seconds`, `external_seconds` (time the metrics took),
and in older rows `lock_wait_seconds` and `label_mtime`.

**top_clusters_by_run.csv, by_run/topk_<ds>.csv**: as above, restricted to the `k_top` (5, 20, 100, 1000) largest
clusters, plus `top_clusters`, `top_molecules`, `top_fraction`, `smallest_top_size`, `scaffold_recall` (per cluster,
share of all molecules with its main scaffold that it holds, size-weighted) and `target_recall`.

**top_clusters.csv, size_bins.csv** (ChEMBL34, SCUBIDOO, Enamine 1M; k = 3, 3.5, 4; order 0). `implementation` as
above (split-merge at P = 36). `top_clusters.csv`: `dataset, implementation, k, seed, rank, size, unique_scaffolds,
isim, label_mtime` for the 20 largest clusters. `size_bins.csv`: `clusters` and `gt0`, `gt1`,
`gt10`, `gt100`, `gt1000` (clusters with more than that many members), `label_mtime` (not written by the current
`top_clusters.py`).

## data/timing/

Each row is one call of `scripts/run/perf_run.py`. Inputs are pre-written in order 0 (split-merge: cut into P
files). All ZINC20 subsets use the 101M threshold (0.2593). We ran them on an otherwise idle machine. Best of 2 up
to 10M rows, a single run from 25M up.

| `implementation` | build | what's timed |
|---|---|---|
| `bblean_serial` | bblean `cpp-tree` a2deb39 | `BitBirch.fit`, one thread |
| `bblean_split_merge` | bblean `cpp-tree` a2deb39 | the whole `bb multiround <dir> --ps P -b 1024 -t T` command: start-up, file reads, all three rounds |
| `bblean_batched` | bblean `cpp-tree-batched` bdad1b3 | `BitBirch(batch_size=4096, n_threads=T).fit` |
| `nvmolkit_batched` | nvMolKit main 763b923 | `bitbirch()` on input already on the GPU, after a warm-up on 100k rows |

**timings.csv**:

| column | meaning |
|---|---|
| implementation, set, n | implementation, set, molecules |
| k, threshold, bf, batch_size | settings |
| threads, processes | CPU threads (`bblean_serial` 1, `bblean_batched` T); P for `bblean_split_merge` |
| repeats, seconds_min, seconds_all | runs, best wall time, all wall times |
| clusters | clusters of the timed run |
| host_peak_gib | peak host memory. `bblean_serial`, `bblean_batched`, `nvmolkit_batched`: peak RSS of the Python process (`getrusage`), including the input (`input_gib`) and for nvMolKit the CUDA runtime. `bblean_split_merge`: bblean's own monitor, summed over its processes, sampled every second. The nvMolKit rows memory-map the input, so mapped pages count too (the `zinc101m` retime row counts the map and a copy); use the GPU columns for nvMolKit. The first CPU row (`bblean_batched`, `chembl34`, 1 thread) may include up to 0.57 GiB of mapped pages |
| input_gib | fingerprint size, N × 256 bytes |
| gpu_pool_peak_gib | `nvmolkit_batched`: peak use of the CUDA default memory pool during the call |
| gpu_input_gib | `nvmolkit_batched`: the device copy of the input, on top of the pool |
| build, commit | build folder (a path on the machine that ran it) and its commit |
| load_before | 1-minute load average at start (info only; newer scripts leave it empty) |
| tag | `sweep` (threads for `bblean_batched`, processes for `bblean_split_merge`: 1, 2, 4, 8, 16, 36, 72 on ChEMBL34 and Enamine 10M); `retime` (GPU rerun alone on the GPU); `other_gpu_procs=N` (older rows only); `rounds={...}` (bblean's per-round seconds for split-merge) |
| when | start time |
| superseded | True for GPU rows timed while other processes held the GPU (`other_gpu_procs` ≥ 1) and later retimed; the figures and `perf_finalize.py` skip them. Empty for rows added after that |

**scaling_exponents.csv**: slope b of log(time) = a + b·log(N) (`time_exponent`) and the same for memory
(`memory_exponent`; host peak for CPU, GPU pool + input for nvMolKit), per `implementation` (`bblean_batched` at 36
threads, `bblean_split_merge` at 36 processes, `bblean_serial`, `nvmolkit_batched`). `scope`: `zinc_10m_101m` (the
four ZINC20 sizes only) or `all_sizes` (every set). Also `points`, `sizes`.

## data/refinement/

Does bblean's own refinement bring every implementation to the same quality? Threshold iSIM + 3.5σ, all five quality
libraries, input orders 0–2 (ChEMBL34 0–9). `implementation`: `bblean_serial` (`cpp-tree` build), `bblean_batched`
(8 threads; stands for nvMolKit too), `bblean_split_merge` (P = 36, tree kept for refinement).

- Serial and batched: yes. After every procedure they're within about 0.01 on iSIM, purity and AMI, cluster counts
  within about 5 %.
- Split-merge: on ChEMBL33 NP refinement brings it level, on ChEMBL35 additions most of the way. On SCUBIDOO,
  Enamine 1M and ChEMBL34 it stays different: 2–6× more clusters, lower scaffold and target AMI (ChEMBL34 after
  `rc0` ×10: 99k clusters vs 33k, scaffold AMI 0.175 vs 0.245, target AMI 0.291 vs 0.365), though its iSIM catches up
  (0.261 vs 0.262).

| `proc` | what | iterations |
|---|---|---|
| `rc0` | `recluster_inplace(iterations=1, extra_threshold=0)` | 10 |
| `rc1` | the same with `extra_threshold=σ` (threshold rises by σ each pass) | 5 |
| `ref` | `set_merge('tolerance-diameter', tolerance=0.05)` then `refine_inplace(X, n_largest=1)` | 5 |

`proc` = `none` or `it` = 0 is the unrefined clustering.

| file | content |
|---|---|
| `quality_<ds>.csv` | one row per order × implementation × procedure × iteration: `dataset, k, seed, implementation, proc, it`, then the quality columns of `quality.csv` |
| `quality_summary.csv` | per library, procedure, iteration (0 and last), implementation: clusters, singletons, largest_cluster, isim, scaffold_purity, medoid_isim_nonsingleton, scaf_ami, tgt_ami, averaged over orders (`n_seeds`) |
| `agreement_<ds>.csv` | AMI/ARI between two labellings (`implementation_a`, `implementation_b`). `kind`: `implementations` (two implementations, same procedure, iteration, order), `input` (refined vs unrefined), `previous` (iteration i vs i−1), `seeds` (orders `seed` and `seed_b`). `rand`, `pair_jaccard`, `pairs_a`, `pairs_b`, `pairs_both` (pair counts), `identical` |
| `summary.csv` | per library and procedure, agreement at iteration 0 and the last: `<a>_vs_<b>_ami/ari/pair_jaccard` (two implementations), `<implementation>_seeds_ami`, `<implementation>_vs_input_ami`, `<implementation>_fixed_point`, `n_seeds` |

Scripts: `scripts/refinement/` (labels go to `BB_REFINE_POST_LABELS`).

## data/moleculeace/

Reruns of the MoleculeACE comparisons of the BitBIRCH paper (Figs. 2, 4, 6) on the 30 sets. Thresholds 0.30–0.80 in
steps of 0.05, branching factor **50**, diameter merge, order 0. `implementation`: `bblean_serial`,
`bblean_split_merge` (`processes` = 10: 10 files and processes, the paper's "10 initial splits"), `nvmolkit_batched`,
`bblean_batched` (indices only).

- `moleculeace_runs.csv`: `set, implementation, processes, n, bf, seed, threshold, seconds` (indicative, not a
  timing), `clusters`.
- `moleculeace_indices.csv`: `set, implementation, processes, n, threshold, clusters, singletons, min_size` (1, 2, 3,
  5, 10: only clusters this big are kept), `clusters_kept`, `chi` (Calinski–Harabasz with Tanimoto, bblean `jt_isim_chi`; higher
  is better), `dbi` (Davies–Bouldin, bblean `jt_dbi`; lower is better), `dunn` (bblean `jt_isim_dunn`, clusters of
  2+ only; higher is better), `dbi_zero_distance_pairs` (cluster pairs with identical centroids; those runs are left
  out of the figures). Code: `scripts/metrics/paper_indices.py`, checked against bblean in `data/checks/`.

## data/zinc22/

Small summaries of the one nvMolKit batched run on all 12,156,333,129 ZINC22 2D molecules (k = 3.5, t = 0.2665,
branching factor 1024). The run itself was done outside these scripts, with a development branch of nvMolKit; the
repo has the data prep (`docs/DATA.md`) and the export of these tables. Written by
`python scripts/data/zinc22_export.py FP_DIR METRICS_DIR [--logs LOG_DIR]`; the figures need only these files.

| file | what |
|---|---|
| `threshold.json` | `n`, `isim` (exact integers), `sigma_n50` (1M random rows), `k`, `threshold` |
| `threshold_overflow_bug.json` | the first, wrong threshold (0.348), from bblean's `jt_isim_from_sum` overflowing (iSIM 0.2024) |
| `run.json` | `n`, `buckets`, `threshold`, `branching_factor`, `batch_size`, `seed`, `readers`, `clusters`, `insert_seconds`, `finalize_seconds`, `wall_seconds`, `gpu_pool_high_gib`, `host_maxrss_gib` |
| `summary.json` | `clusters`, `singletons`, `largest`, `molecules_in_top` (10 / 100 / 1,000 / 10,000 largest), `size_quantiles` (50 / 90 / 99 / 99.9 %), `clusters_ge` (clusters with at least 2 … 100,000 members) |
| `run_progress.csv` | per input part (128): `bucket, rows, rows_total, read_s, wait_s, lock_wait_s, insert_s, clusters, gpu_pool_used_gib, gpu_pool_high_gib, host_rss_gib, elapsed_s` |
| `failed_run_progress.csv` | same columns, first run at the wrong threshold, stopped after 4 parts (out of GPU memory) |
| `largest_cluster_sizes.csv` | `rank, size` of the 10,000 largest clusters |
| `fp_metrics.json` | `n_molecules, clusters, singletons, largest_cluster, isim, library_isim, medoid_isim_all, medoid_isim_nonsingleton, pass1_s, pass2_s, gpu_peak_gib` |
| `zinc22_overall.csv` | the above plus scaffold measures on a 500M sample: `scaffold_purity_pooled_plugin`, `scaffold_purity_m100_plugin` / `_split` (clusters with 100+ sampled members; the two bracket the true value), `scaffold_purity_m100_cover`, `scaf_ari`, `scaf_ami`, `scaf_v_measure`, and the same three on 50M (`_50M`) |
| `zinc22_topk.csv` | per k_top = 5, 20, 100, 1000: `top_molecules, top_fraction, smallest_top_size, isim, isim_min, isim_max, medoid_isim_topk, sampled_members, min_sampled, scaffold_purity` (+ `_se`, split value), `scaffold_recall` (+ `_se`), `scaf_ari, scaf_ami, scaf_v_measure` |
| `shard_stats.json` | `files`, `failed_smiles`, `max_column_sum` and its square / 2^64 (why 64-bit sums overflow) |
| `log_values.json` | `download_seconds`, `download_bytes`, `fingerprint_mol_per_s`, `route_seconds`, each with `value`, `log`, `pattern`, `line` |

AMI and V-measure on a sample shift with sample size, so only ARI compares with the smaller libraries.

## data/thresholds/

`isim_sigma.json`, one entry per set (`chembl34`, `scu`, `ena1m`, `c35`, `np`, `ena10m`, `zinc101m`; the ZINC20
subsets use `zinc101m`). Every script reads thresholds from here; a new results folder without its own file uses
this one. Keys: `n`, `isim` (exact, `bblean.similarity.jt_isim_packed`), `sigma_n50` (`estimate_jt_std(n_samples=50)`,
used for the thresholds; deterministic), `sigma_default` (bblean's default sample count, for comparison; not for
`zinc101m`), `t_3`, `t_3.5`, `t_4`, `secs`, `source` (`zinc101m` only). Made by `scripts/metrics/thresholds.py`
(`zinc101m` by `scripts/data/isim_sigma_streamed.py`).

## data/checks/

Runs that must give identical clusters do. "Same partition" = same grouping, cluster numbers may differ. Some
columns hold paths on the machine that ran them, as a record. "Earlier labels" = an earlier run with the same
settings (bblean 0.10.3 and an nvMolKit development build).

| file | checks | columns |
|---|---|---|
| `build_identity_checks.csv` | each final build vs the earlier labels it replaces (c35, np, ChEMBL34, k = 3.5, order 0; bblean serial against bblean 0.10.3, the batched ones against the earlier nvMolKit build); also split-merge on the C++ tree vs on bblean 0.10.3 (`build` = `bblean 0.10.3`: same 272,185 clusters, 4.4× slower) and nvMolKit at batch size 1024 (a different partition, as expected) | `dataset, implementation, processes, build, k, seed, threshold, bf, batch_size, threads, seconds, clusters, same_partition_as_earlier_labels` |
| `bblean_serial_cpp_tree_vs_bblean_0.10.3.csv` | bblean serial `cpp-tree` vs bblean 0.10.3, every library × k × order (57 labellings): all identical | `dataset, k, seed, cpp_tree_labels, bblean_0.10.3_labels, same_partition, clusters_cpp_tree, clusters_bblean_0.10.3, cpp_tree_build, cpp_tree_seconds` |
| `bblean_batched_vs_nvmolkit.csv` | bblean batched vs nvMolKit batched for every library, k and input order with labels from both (64 pairs): all identical. `nvmolkit_build`: 763b923 (ChEMBL34 orders 3–9 at k = 3.5) or the earlier development build (the rest). Recomputed from the stored labels; new runs add rows with `scripts/run/check_batched_vs_nvmolkit.py` | `dataset, k, seed, clusters_bblean_batched, clusters_nvmolkit_batched, nvmolkit_build, same_partition` |
| `refinement_rerun_identity.csv` | refinement rerun with the final builds vs its first run: all 945 label files identical | `dataset, implementation, proc, file, old_exists, same, clusters_new, clusters_old` |
| `paper_indices_vs_bblean.csv` | GPU Calinski–Harabasz / Davies–Bouldin / Dunn vs bblean's own, CHEMBL233_Ki, t = 0.45 / 0.55 / 0.65: relative difference ≤ 4e-6 | `set, threshold, bf, min_size, clusters_kept, index, bblean, gpu, rel_diff` |
