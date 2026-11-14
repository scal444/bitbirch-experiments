#!/bin/bash
# Every untimed run behind the quality, MoleculeACE and refinement results, in order. Tables go to data/ of the
# results folder (results_<BB_RESULTS_NAME>, config/paths.env); labels go to BB_OUT.
#
#   bash scripts/run/run_quality_all.sh [STAGE ...]      (default: all stages, in this order)
#
# Stages:
#   labels   labels of the four implementations for every dataset x k {3, 3.5, 4} x input order, ChEMBL34 orders 3-9
#            at k = 3.5, split-merge at P = 10 / 72 on two sets; then the check that bblean batched and nvMolKit
#            batched give the same partitions
#   metrics  quality metrics per run, the combined tables, the largest-cluster tables, and the agreement between
#            runs on different input orders, within and across implementations (data/quality/agreement.csv)
#   moleculeace  MoleculeACE runs (30 sets x 11 thresholds) and their CH / DB / Dunn indices
#   refine   the refinement experiment (scripts/refinement): runs, analysis, summary, working figures
#
# Run from any directory inside the Python environment from setup/. Paths and builds come from config/paths.env
# (BB_DATA, BB_OUT, BB_BBLEAN_TREE, BB_BBLEAN_BATCHED, BB_NVMOLKIT_PY, BB_REFINE_POST_LABELS, ...). Needs a GPU for
# nvMolKit and the metrics. Every step skips outputs that already exist, so the script can be rerun after
# a stop. Long: run it detached, e.g. setsid bash scripts/run/run_quality_all.sh < /dev/null > quality.log 2>&1 &
#
# Machine-specific settings (environment variables):
THREADS=${THREADS:-36}     # CPU threads for bblean batched
P_MAIN=${P_MAIN:-36}       # processes (= input parts) for bblean split-merge

cd "$(dirname "$0")/../.."
DATASETS="chembl34 scu ena1m c35 np"
EXTRA_ORDERS="3 4 5 6 7 8 9"   # extra ChEMBL34 input orders at k = 3.5 (orders 0-2 come with the defaults)
R=scripts/run
run() { echo "+ $*"; "$@" || echo "FAILED: $*"; }

stage_labels() {
  for ds in $DATASETS; do
    run python $R/run_bblean.py bblean_serial "$ds"
    run python $R/run_bblean.py bblean_batched "$ds" --threads "$THREADS"
    run python $R/run_nvmolkit.py "$ds"
    run python $R/run_bblean_multiround.py "$ds" --files "$P_MAIN" --ps "$P_MAIN"
  done
  run python $R/run_bblean.py bblean_serial chembl34 3.5 --seeds $EXTRA_ORDERS
  run python $R/run_bblean.py bblean_batched chembl34 3.5 --seeds $EXTRA_ORDERS --threads "$THREADS"
  run python $R/run_nvmolkit.py chembl34 3.5 --seeds $EXTRA_ORDERS
  run python $R/run_bblean_multiround.py chembl34 3.5 --seeds $EXTRA_ORDERS --files "$P_MAIN" --ps "$P_MAIN"
  for P in 10 72; do for ds in chembl34 scu; do
    run python $R/run_bblean_multiround.py "$ds" 3.5 --seeds 0 --files "$P" --ps "$P"
  done; done
  for ds in chembl34 scu ena1m; do for k in 3 3.5 4; do run python $R/check_batched_vs_nvmolkit.py "$ds" "$k" 0 1 2; done; done
  for ds in c35 np; do for k in 3 3.5 4; do run python $R/check_batched_vs_nvmolkit.py "$ds" "$k" 0 1 2 3 4; done; done
  run python $R/check_batched_vs_nvmolkit.py chembl34 3.5 $EXTRA_ORDERS
}

stage_metrics() {
  run python scripts/metrics/metrics.py $DATASETS
  run python scripts/metrics/combine_quality.py
  run python scripts/metrics/top_clusters.py
  run python scripts/metrics/agreement.py $DATASETS
}

stage_moleculeace() {
  run python $R/run_moleculeace.py bblean_serial bblean_split_merge nvmolkit_batched
  run python $R/run_moleculeace.py bblean_batched
  run python scripts/metrics/moleculeace_metrics.py
}

stage_refine() {
  local rp=scripts/refinement
  for ds in $DATASETS; do for s in 0 1 2; do
    run python $rp/run_refine.py "$ds" --seed "$s" --implementations bblean_serial,bblean_split_merge
    run python $rp/run_refine.py "$ds" --seed "$s" --implementations bblean_batched
  done; done
  for s in $EXTRA_ORDERS; do
    run python $rp/run_refine.py chembl34 --seed "$s" --implementations bblean_serial,bblean_split_merge
    run python $rp/run_refine.py chembl34 --seed "$s" --implementations bblean_batched
  done
  run python $rp/analyze.py $DATASETS --seeds 0,1,2 --quality-seeds 0,1,2
  run python $rp/analyze.py chembl34 --seeds 0,1,2,3,4,5,6,7,8,9 --quality-seeds 0,1,2,3,4,5,6,7,8,9
  run python $rp/summarize.py
  run python $rp/plot.py
}

STAGES=${*:-labels metrics moleculeace refine}
for st in $STAGES; do
  date; echo "STAGE $st"
  "stage_$st"
done
date; echo QUALITY_ALL_DONE
