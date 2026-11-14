"""One quality table for all implementations, from the per-run metrics CSVs of metrics.py.

   python scripts/metrics/combine_quality.py  -> data/quality/quality.csv, data/quality/top_clusters_by_run.csv

Reads data/quality/by_run/*.csv and, where a results folder has it, data/quality/by_run_earlier_labels/*.csv (metrics
of labels from an earlier run that give the same partitions; see the results folder README). Where both have a row
for the same implementation, processes, bf, k and input order, the by_run/ row is kept.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

ID = ["dataset", "implementation", "processes", "bf", "k", "seed"]
KEEP = ID + ["n_molecules", "threshold", "seconds", "clusters", "singletons", "singleton_fraction",
             "singletons_per_cluster", "largest_cluster", "nonsingleton_clusters", "isim", "library_isim",
             "scaffold_purity", "medoid_isim_all", "medoid_isim_nonsingleton", "tgt_molecules"] + \
    [f"{r}_{m}" for r in ("scaf", "tgt") for m in ("ari", "ami", "v_measure", "homogeneity", "completeness")]
BOOKKEEPING = ("label_mtime", "metrics_seconds", "gpu_seconds", "external_seconds")

for name in ("", "topk_"):
    frames = []
    for root in (P.QUALITY / "by_run", P.QUALITY / "by_run_earlier_labels"):  # by_run first: it wins on duplicates
        for f in sorted(root.glob(f"{name}*.csv")):
            if not name and f.name.startswith("topk_"):
                continue
            frames.append(pd.read_csv(f, dtype={"processes": "Int64"}))
    if not frames:
        print("no per-run metrics under", P.QUALITY)
        continue
    df = pd.concat(frames, ignore_index=True)
    key = ID + (["k_top"] if name else [])
    df = df[~df[key].astype(str).duplicated(keep="first")]
    cols = [c for c in KEEP if c in df] if not name else \
        ID + [c for c in df.columns if c not in ID and c not in BOOKKEEPING and c != "seconds"]
    out = P.QUALITY / ("top_clusters_by_run.csv" if name else "quality.csv")
    df[cols].sort_values(key).to_csv(out, index=False)
    print(out, len(df))
