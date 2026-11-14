"""Export the small files the figures read about the ZINC22 run into data/zinc22/ of the results folder.

  python scripts/data/zinc22_export.py FP_DIR METRICS_DIR [--logs LOG_DIR] [--out DIR]

The ZINC22 clustering (12.16 billion molecules, nvMolKit batched, development build `bitbirch-gpu-perf`) produced
folders far too large to ship with this repository. This script reads them and writes the few small files the
figure scripts need, so those scripts run from the repository alone.

Inputs
  FP_DIR        the fingerprint folder made by zinc22_fingerprint.py / zinc22_buckets.py:
                  threshold.json, threshold.overflow_bug_0.348.json (the first, wrong threshold),
                  cluster_k3.5_bf1024/{run.json, summary.json, progress.csv}       (the clustering run),
                  cluster_k3.5_bf1024.failed_overflow_threshold_0.348/progress.csv (the stopped run),
                  shards/*.done and *.colsum                                       (one pair per input file)
  METRICS_DIR   the quality metrics of the run: zinc22_overall.csv, zinc22_topk.csv, fp_metrics.json,
                clusters_fp.npy (one record per cluster, with a "size" field)
  LOG_DIR       the run's log folder (default METRICS_DIR/../../logs): download_z22.log, z22_ingest.log,
                z22_route.log

Outputs (see the results folder README): copies of the small json/csv files (with file paths removed from
run.json), shard_stats.json, log_values.json and largest_cluster_sizes.csv.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.common import paths as P  # noqa: E402

TOP = 10_000
RUN = "cluster_k3.5_bf1024"
FAILED = "cluster_k3.5_bf1024.failed_overflow_threshold_0.348"
# values read from the logs: name -> (log file, regular expression; the last match is used)
LOG_VALUES = {
    "download_seconds": ("download_z22.log", r"done in (\d+)s"),
    "download_bytes": ("download_z22.log", r"done in \d+s, (\d+) bytes"),
    "fingerprint_mol_per_s": ("z22_ingest.log", r"([\d,]+) mol/s\n"),
    "route_seconds": ("z22_route.log", r"this run, (\d+)s"),
}


def log_value(path, pat):
    text = path.read_text(errors="replace")
    hits = list(re.finditer(pat, text))
    if not hits:
        return None
    m = hits[-1]
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.start())
    line = text[start:end if end >= 0 else len(text)]
    v = m.group(1).replace(",", "")
    return {"value": int(v) if v.isdigit() else float(v), "log": path.name, "pattern": pat, "line": line}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fp_dir", type=Path)
    ap.add_argument("metrics_dir", type=Path)
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=P.ZINC22)
    a = ap.parse_args()
    fp, met, out = a.fp_dir.expanduser(), a.metrics_dir.expanduser(), a.out
    logs = (a.logs or met.parent.parent / "logs").expanduser()
    out.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(fp / "threshold.json", out / "threshold.json")
    shutil.copyfile(fp / "threshold.overflow_bug_0.348.json", out / "threshold_overflow_bug.json")
    rj = json.loads((fp / RUN / "run.json").read_text())
    rj = {k: v for k, v in rj.items() if not (isinstance(v, str) and "/" in v)}  # drop file paths
    (out / "run.json").write_text(json.dumps(rj, indent=1) + "\n")
    shutil.copyfile(fp / RUN / "summary.json", out / "summary.json")
    shutil.copyfile(fp / RUN / "progress.csv", out / "run_progress.csv")
    shutil.copyfile(fp / FAILED / "progress.csv", out / "failed_run_progress.csv")
    for name in ("zinc22_overall.csv", "zinc22_topk.csv", "fp_metrics.json"):
        shutil.copyfile(met / name, out / name)

    # input files, failed SMILES and the largest per-bit column sum (whose square overflows uint64)
    sh = fp / "shards"
    infos = [json.loads(p.read_text()) for p in sorted(sh.glob("*.done"))]
    cs = sum(np.fromfile(sh / f"{i['file_index']:05d}.colsum", np.uint64).astype(object) for i in infos)
    mx = int(max(cs))
    (out / "shard_stats.json").write_text(json.dumps(
        {"files": len(infos), "failed_smiles": int(sum(i["failed"] for i in infos)), "max_column_sum": mx,
         "max_column_sum_squared_over_2_64": mx * mx / 2 ** 64}, indent=1) + "\n")

    lv = {}
    for name, (log, pat) in LOG_VALUES.items():
        p = logs / log
        lv[name] = log_value(p, pat) if p.exists() else None
    (out / "log_values.json").write_text(json.dumps(lv, indent=1) + "\n")

    sizes = np.sort(np.load(met / "clusters_fp.npy", mmap_mode="r")["size"])[::-1][:TOP]
    with open(out / "largest_cluster_sizes.csv", "w") as fh:
        fh.write("rank,size\n")
        for i, s in enumerate(sizes, 1):
            fh.write(f"{i},{int(s)}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
