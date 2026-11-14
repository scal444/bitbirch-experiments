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
#   big       the same three at 50M and 101M rows (edit stage_big to drop sizes your machine has no memory for)
#   finalize  scripts/run/perf_finalize.py --write (data/timing/scaling_exponents.csv)
#
# Run inside the Python environment from setup/, with paths and builds from config/paths.env (BB_DATA, BB_OUT,
# BB_BBLEAN_TREE, BB_BBLEAN_BATCHED, BB_NVMOLKIT_PY). Run it on an otherwise idle machine. Stages take hours at the
# large sizes: run detached, e.g. setsid bash scripts/run/run_timings.sh < /dev/null > timings.log 2>&1 &
#
# Machine-specific settings (environment variables):
THREADS=${THREADS:-36}                    # CPU threads for bblean batched at every size
P_MAIN=${P_MAIN:-36}                      # bb multiround processes (= input parts) at every size
WORKERS=${WORKERS:-"1 2 4 8 16 36 72"}    # thread / process counts of the sweep

cd "$(dirname "$0")/../.."
[ -f config/paths.env ] || [ -n "$BB_RESULTS_NAME" ] || {
  echo "config/paths.env not found: run \`cp config/paths.example.env config/paths.env\` and edit it"; exit 1; }
SIZES="ena1m chembl34 ena10m zinc10m zinc25m zinc50m zinc101m"
CPU_SIZES="ena1m chembl34 ena10m zinc10m zinc25m"
SWEEP_SETS="chembl34 ena10m"

FAILED=()   # failed runs; the script keeps going and lists them at the end
pr() {  # one timed run
  python scripts/run/perf_run.py "$@" || { echo "FAILED perf_run $*"; FAILED+=("perf_run $*"); }
}
reps() { case $1 in zinc25m|zinc50m|zinc101m) echo 1;; *) echo 2;; esac; }

stage_inputs() {
  python scripts/run/perf_inputs.py $SWEEP_SETS --parts $WORKERS || exit 1
  python scripts/run/perf_inputs.py $SIZES --parts "$P_MAIN" || exit 1
}

stage_gpu() {
  for s in $SIZES; do pr nvmolkit_batched "$s" --repeats 2; done
}

stage_sweep() {
  for s in $SWEEP_SETS; do
    for T in $WORKERS; do pr bblean_batched "$s" --threads "$T" --repeats 2 --tag sweep; done
    for P in $WORKERS; do pr bblean_split_merge "$s" --ps "$P" --repeats 2 --tag sweep; done
  done
}

stage_cpu() {
  for s in $CPU_SIZES; do
    case " $SWEEP_SETS " in *" $s "*) ;; *) pr bblean_batched "$s" --threads "$THREADS" --repeats "$(reps "$s")";; esac
  done
  for s in $CPU_SIZES; do pr bblean_split_merge "$s" --ps "$P_MAIN" --repeats "$(reps "$s")"; done
  for s in $CPU_SIZES; do pr bblean_serial "$s" --repeats "$(reps "$s")"; done
}

stage_big() {
  pr bblean_batched zinc50m --threads "$THREADS" --repeats 1
  pr bblean_split_merge zinc50m --ps "$P_MAIN" --repeats 1
  pr bblean_serial zinc50m --repeats 1
  pr bblean_batched zinc101m --threads "$THREADS" --repeats 1
  pr bblean_split_merge zinc101m --ps "$P_MAIN" --repeats 1
  pr bblean_serial zinc101m --repeats 1
}

stage_finalize() { python scripts/run/perf_finalize.py --write || FAILED+=("perf_finalize"); }

STAGES=${*:-inputs gpu sweep cpu big finalize}
for st in $STAGES; do
  date; echo "STAGE $st"
  "stage_$st"
done
date
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "TIMINGS_DONE with ${#FAILED[@]} failed run(s):"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
echo TIMINGS_DONE
