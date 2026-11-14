"""Per-cluster numbers of the largest clusters (cheap, CPU, from existing labels), in the form of the BitBIRCH best-practices
paper's Fig 5 / S17.

  python scripts/metrics/top_clusters.py [DS ...] [--force]       (default: chembl34 scu ena1m)

For each implementation in the quality comparison, k in {3, 3.5, 4}, input order 0, bf 1024:
  top_clusters.csv : the 20 largest clusters (ties: lowest first member row): size, unique Murcko scaffolds, iSIM
  size_bins.csv    : clusters with more than 0 / 1 / 10 / 100 / 1000 members (as in the paper's right-hand panels)
Both under data/quality/ of the results folder. Rows already in the tables are kept; --force recomputes them.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

MAIN_P = 36  # processes of the bblean split-merge runs in the comparison
IMPLS = {  # implementation -> label folder under BB_OUT/labels
    "serial": "bblean_tree",  # bblean serial
    "batched": "nvmolkit_main",  # nvMolKit batched (GPU)
    "multiround": f"bblean_mr{MAIN_P}",  # bblean split-merge
    "bblean_batched": "bblean_batched",  # bblean batched (multithreaded CPU)
}
OUT = P.QUALITY


TOP = 20
BINS = (0, 1, 10, 100, 1000)


def isim_rows(fps, rows):
    bits = np.unpackbits(fps[np.sort(rows)], axis=1).astype(np.int64)
    s = bits.sum(0)
    n = len(rows)
    kq, sq = s.sum(), (s * s).sum()
    a = (sq - kq) / 2.0
    return float(a / ((a + n * kq) - sq)) if kq else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["chembl34", "scu", "ena1m"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tf, bf_ = OUT / "top_clusters.csv", OUT / "size_bins.csv"
    top = pd.read_csv(tf) if tf.exists() else pd.DataFrame()
    bins = pd.read_csv(bf_) if bf_.exists() else pd.DataFrame()
    for ds in a.datasets:
        fps = scaf = None
        for pl, method in IMPLS.items():
            for k in P.KS:
                f = P.label_path(ds, method, k, 0)
                if not (f.exists() and f.with_suffix(".json").exists()):
                    continue
                if not a.force and len(top) and ((top.dataset == ds) & (top.player == pl) & (top.k == k)).any():
                    continue
                if fps is None:
                    fps = np.load(P.DATASETS[ds]["fps"], mmap_mode="r")
                    scaf = np.load(P.DATASETS[ds]["scaf"])
                lab = np.load(f)
                _, first, inv, sizes = np.unique(lab, return_index=True, return_inverse=True, return_counts=True)
                order = np.lexsort((first, -sizes))[:TOP]
                rows = []
                for rank, c in enumerate(order, 1):
                    mem = np.flatnonzero(inv == c)
                    rows.append(dict(dataset=ds, player=pl, k=k, seed=0, rank=rank, size=int(sizes[c]),
                                     unique_scaffolds=int(len(np.unique(scaf[mem]))), isim=isim_rows(fps, mem)))
                brow = dict(dataset=ds, player=pl, k=k, seed=0, clusters=len(sizes))
                brow.update({f"gt{b}": int((sizes > b).sum()) for b in BINS})
                sel = lambda d: (d.dataset == ds) & (d.player == pl) & (d.k == k)  # noqa: E731
                top = pd.concat([top[~sel(top)] if len(top) else top, pd.DataFrame(rows)], ignore_index=True)
                bins = pd.concat([bins[~sel(bins)] if len(bins) else bins, pd.DataFrame([brow])],
                                 ignore_index=True)
                for d, p in ((top, tf), (bins, bf_)):
                    d.sort_values([c for c in ("dataset", "player", "k", "rank") if c in d]).to_csv(
                        p.with_suffix(".tmp"), index=False)
                    os.replace(p.with_suffix(".tmp"), p)
                print(ds, pl, f"k{k:g}", "top sizes", [r["size"] for r in rows[:5]], flush=True)


if __name__ == "__main__":
    main()
