#!/bin/bash
# Every timed run behind the speed and memory figures (data/timing/timings.csv of the results folder,
# results_<BB_RESULTS_NAME>), then the scaling fits.
#
#   bash scripts/run/run_timings.sh [STAGE ...]      (default: inputs gpu sweep cpu big finalize)
#
# Stages:
#   inputs    shuffled inputs (seed-0 order) and the pre-split bb multiround input dirs (untimed)
#   gpu       nvMolKit batched (GPU) on every size
#   sweep     bblean batched against threads and bblean split-merge (bb multiround) against processes,
#             on ChEMBL34 and Enamine 10M
#   cpu       bblean batched (THREADS threads), bblean split-merge (P_MAIN processes) and bblean serial, up to 25M rows
#   big       the same three at 50M and 101M rows; 101M is skipped for an implementation whose 50M peak memory
#             times 2.2 would exceed HOST_MEM_GIB
#   finalize  scripts/run/perf_finalize.py --write (superseded flags, data/timing/scaling_exponents.csv)
#
# Run inside the Python environment from setup/, with paths and builds from config/paths.env (BB_DATA, BB_OUT,
# BB_BBLEAN_TREE, BB_BBLEAN_BATCHED, BB_NVMOLKIT_PY). Run it on an otherwise idle machine. Stages take hours at the
# large sizes: run detached, e.g. setsid bash scripts/run/run_timings.sh < /dev/null > timings.log 2>&1 &
#
# Machine-specific settings (environment variables):
THREADS=${THREADS:-36}                    # CPU threads for bblean batched at every size
P_MAIN=${P_MAIN:-36}                      # bb multiround processes (= input parts) at every size
WORKERS=${WORKERS:-"1 2 4 8 16 36 72"}    # thread / process counts of the sweep
HOST_MEM_GIB=${HOST_MEM_GIB:-400}         # host RAM available to one run

cd "$(dirname "$0")/../.."
SIZES="ena1m chembl34 ena10m zinc10m zinc25m zinc50m zinc101m"
CPU_SIZES="ena1m chembl34 ena10m zinc10m zinc25m"
SWEEP_SETS="chembl34 ena10m"

pr() {  # one timed run
  python scripts/run/perf_run.py "$@" || echo "FAILED perf_run $*"
}
reps() { case $1 in zinc25m|zinc50m|zinc101m) echo 1;; *) echo 2;; esac; }
fits() {  # does the 101M run of this implementation fit in HOST_MEM_GIB, judged from its 50M peak?
  python - "$1" "$HOST_MEM_GIB" <<'PY'
import sys, pandas as pd
from scripts.common import paths as P
d = pd.read_csv(P.TIMING / "timings.csv"); d = d[(d.player == sys.argv[1]) & (d.set == "zinc50m")]
sys.exit(0 if len(d) == 0 or float(d.host_peak_gib.max()) * 2.2 < float(sys.argv[2]) else 1)
PY
}

stage_inputs() {
  python scripts/run/perf_inputs.py $SWEEP_SETS --parts $WORKERS || exit 1
  python scripts/run/perf_inputs.py $SIZES --parts "$P_MAIN" || exit 1
}

stage_gpu() {
  for s in $SIZES; do pr nvmolkit "$s" --repeats 2; done
}

stage_sweep() {
  for s in $SWEEP_SETS; do
    for T in $WORKERS; do pr batched "$s" --threads "$T" --repeats 2 --tag sweep; done
    for P in $WORKERS; do pr multiround "$s" --ps "$P" --repeats 2 --tag sweep; done
  done
}

stage_cpu() {
  for s in $CPU_SIZES; do
    case " $SWEEP_SETS " in *" $s "*) ;; *) pr batched "$s" --threads "$THREADS" --repeats "$(reps "$s")";; esac
  done
  for s in $CPU_SIZES; do pr multiround "$s" --ps "$P_MAIN" --repeats "$(reps "$s")"; done
  for s in $CPU_SIZES; do pr serial "$s" --repeats "$(reps "$s")"; done
}

stage_big() {
  pr batched zinc50m --threads "$THREADS" --repeats 1
  pr multiround zinc50m --ps "$P_MAIN" --repeats 1
  pr serial zinc50m --repeats 1
  if fits batched; then pr batched zinc101m --threads "$THREADS" --repeats 1; else echo "SKIP batched zinc101m (memory)"; fi
  if fits multiround; then pr multiround zinc101m --ps "$P_MAIN" --repeats 1; else echo "SKIP multiround zinc101m (memory)"; fi
  if fits serial; then pr serial zinc101m --repeats 1; else echo "SKIP serial zinc101m (memory)"; fi
}

stage_finalize() { python scripts/run/perf_finalize.py --write; }

STAGES=${*:-inputs gpu sweep cpu big finalize}
for st in $STAGES; do
  date; echo "STAGE $st"
  "stage_$st"
done
date; echo TIMINGS_DONE
