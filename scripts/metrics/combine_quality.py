"""One quality table for all implementations, from the per-run metrics CSVs of metrics.py.

   python scripts/metrics/combine_quality.py  -> data/quality/quality.csv, data/quality/top_clusters_by_run.csv

Reads data/quality/by_run/*.csv and the metrics of the earlier labels in data/quality/by_run_earlier_labels/
(all in the results folder). Rows of bblean's
refinement (bblean_rc0/rc1, nvmolkit_refined) are left out. Serial rows come from `bblean_tree`; earlier `bblean`
rows (same partitions) are dropped wherever a `bblean_tree` row exists. The `player` column numbers the
implementations: 1 bblean serial, 2 bblean split-merge (bb multiround) P<n>, "3=4 batched" nvMolKit batched (GPU)
and bblean batched (multithreaded CPU), which give the same partitions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

DROP = {"bblean_rc0", "bblean_rc1", "nvmolkit_refined"}
PLAYER = {"bblean": "1 bblean serial", "bblean_tree": "1 bblean serial", "nvmolkit": "3=4 batched",
          "nvmolkit_main": "3=4 batched", "bblean_batched": "3=4 batched", "butina_same": "Butina (same t)",
          "butina_matched": "Butina (count-matched)"}
KEEP = ["dataset", "player", "method", "base_method", "bf", "k", "seed", "source", "n_molecules", "threshold",
        "seconds", "clusters", "singletons", "singleton_fraction", "singletons_per_cluster", "largest_cluster",
        "nonsingleton_clusters", "isim", "library_isim", "scaffold_purity", "medoid_isim_all",
        "medoid_isim_nonsingleton", "tgt_molecules"] + [f"{r}_{m}" for r in ("scaf", "tgt", "butina")
                                                         for m in ("ari", "ami", "v_measure", "homogeneity",
                                                                   "completeness")]


def player(m):
    b = m.split("_bf")[0]
    if b.startswith("bblean_mr"):
        return f"2 bblean multiround P{b[9:]}"
    return PLAYER.get(b, b)


for name, keep in (("", KEEP), ("topk_", None)):
    frames = []
    for src, root in (("bp", P.QUALITY / "by_run_earlier_labels"), ("techreport", P.QUALITY / "by_run")):
        for f in sorted(root.glob(f"{name}*.csv")):
            if not name and f.name.startswith("topk_"):
                continue
            d = pd.read_csv(f)
            d = d[~d.method.isin(DROP)].copy()
            d["source"] = src
            frames.append(d)
    if not frames:
        print("no per-run metrics under", P.QUALITY)
        continue
    df = pd.concat(frames, ignore_index=True)
    fork = set(
        map(tuple, df.loc[df.method == "bblean_tree", ["dataset", "k", "seed"]].values.tolist()))
    old = (df.method == "bblean").to_numpy() & np.array([t in fork for t in map(tuple, df[["dataset", "k", "seed"]].values.tolist())], bool)
    df = df[~old]
    df["player"] = df.method.map(player)
    cols = keep or [c for c in df.columns if not c.endswith("seconds") and c not in ("label_mtime", "butina_used")]
    if keep is None:
        cols = ["dataset", "player"] + [c for c in cols if c not in ("dataset", "player")]
    out = P.QUALITY / ("top_clusters_by_run.csv" if name else "quality.csv")
    df[cols].sort_values(["dataset", "method", "k", "seed"] + (["k_top"] if name else [])).to_csv(out, index=False)
    print(out, len(df))
