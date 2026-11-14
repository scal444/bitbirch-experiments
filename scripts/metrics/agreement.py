"""Agreement (AMI, ARI) between the cluster memberships of two runs, within and across implementations.

  python scripts/metrics/agreement.py [DS ...] [--ks 3 3.5 4] [--force]

This is how the BitBIRCH authors judge a new implementation: compare its clusters with those of a regular bblean
serial run, and set that against how much bblean serial disagrees with itself when only the input order changes.
bblean serial is not ground truth: two of its runs on different input orders already differ, so that agreement is
the floor (the most a new implementation can be expected to reach).

Pairs, per library and k (branching factor 1024); every pair uses two DIFFERENT input orders:
  within an implementation X : X on order a against X on order b, every unordered pair a < b
  across implementations     : bblean serial on order a against X on order b, every ordered pair with a != b,
                               X = bblean split-merge, bblean batched, nvMolKit batched
Every input order with labels for the implementation is used.

Subsets (column `subset`):
  all    every molecule (singletons included)
  top20  the molecules in the 20 largest clusters of the reference run; AMI/ARI of both runs restricted to them
  top50  the same with the 50 largest clusters
The reference run is the first run of the pair (bblean serial for pairs across implementations). Within an
implementation neither run comes first, so the value is the mean of the two choices (each run as reference once);
n_molecules, clusters_a and clusters_b are then means as well. Ties in cluster size are broken by the lower cluster id.

Writes data/quality/agreement.csv of the results folder (columns: dataset, k, implementation_a, implementation_b,
order_a, order_b, subset, ami, ari, n_molecules, clusters_a, clusters_b). Pairs already in the table are kept; --force
recomputes all. Implementations: bblean_serial, bblean_split_merge (36 processes), bblean_batched, nvmolkit_batched
(scripts/common/implementations.py). AMI/ARI from ext_metrics (GPU, or multithreaded CPU torch; matches
scikit-learn, see validate_metrics.py).
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import IMPLEMENTATIONS, label_folder  # noqa: E402
from scripts.metrics.ext_metrics import external_metrics  # noqa: E402

REF = "bblean_serial"
TOPS = (20, 50)
OUTF = P.QUALITY / "agreement.csv"
KEY = ["dataset", "k", "implementation_a", "implementation_b", "order_a", "order_b", "subset"]
COLS = KEY + ["ami", "ari", "n_molecules", "clusters_a", "clusters_b"]


def orders_with_labels(ds, impl, k):
    """Every input order that has labels for this implementation: {order: label file s<order>.npy}."""
    d = P.label_path(ds, label_folder(impl), k, 0).parent
    found = {int(f.stem[1:]): f for f in d.glob("s*.npy") if f.stem[1:].isdigit()} if d.is_dir() else {}
    return dict(sorted(found.items()))


def top_mask(ref, top):
    """Molecules in the `top` largest clusters of `ref` (ties: lower cluster id first)."""
    _, inv, sz = np.unique(ref, return_inverse=True, return_counts=True)
    keep = np.zeros(len(sz), bool)
    keep[np.lexsort((np.arange(len(sz)), -sz))[:top]] = True
    return keep[inv.ravel()]


def metrics(la, lb, subset=None):
    m = external_metrics(lb, la, subset=subset)  # ref = run a
    return dict(ami=m["ami"], ari=m["ari"], n_molecules=m["n"], clusters_a=m["n_ref"], clusters_b=m["n_clusters"])


def pair_rows(la, lb, symmetric):
    """{subset: metrics} for one pair; top-k averaged over both reference choices when symmetric."""
    out = {"all": metrics(la, lb)}
    for t in TOPS:
        m = metrics(la, lb, top_mask(la, t))
        if symmetric:
            r = metrics(lb, la, top_mask(lb, t))  # b as reference: swap the roles back for clusters_a / _b
            r["clusters_a"], r["clusters_b"] = r["clusters_b"], r["clusters_a"]
            m = {c: (m[c] + r[c]) / 2 for c in m}
        out[f"top{t}"] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(P.DATASETS))
    ap.add_argument("--ks", nargs="*", type=float, default=list(P.KS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    old = pd.read_csv(OUTF) if OUTF.exists() and not a.force else pd.DataFrame(columns=COLS)
    rows = old.to_dict("records")
    done = {tuple(r[c] for c in KEY[:6]) for r in rows}
    cache = {}

    def load(p):
        if p not in cache:
            if len(cache) > 24:
                cache.clear()
            cache[p] = np.load(p).ravel()
        return cache[p]

    def flush():
        OUTF.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(rows, columns=COLS).sort_values(KEY)
        tmp = OUTF.with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, OUTF)

    for ds in a.datasets:
        for k in a.ks:
            files = {impl: orders_with_labels(ds, impl, k) for impl in IMPLEMENTATIONS}
            jobs = []
            for impl, fs in files.items():  # within: unordered pairs of different orders
                jobs += [(impl, impl, sa, sb) for sa, sb in itertools.combinations(fs, 2)]
            for impl in IMPLEMENTATIONS:  # across: serial on order a against impl on order b, a != b
                if impl != REF:
                    jobs += [(REF, impl, sa, sb) for sa in files[REF] for sb in files[impl] if sa != sb]
            print(ds, f"k{k:g}", {i: list(f) for i, f in files.items()}, f"{len(jobs)} pairs", flush=True)
            new = 0
            for ia, ib, sa, sb in jobs:
                if (ds, k, ia, ib, sa, sb) in done:
                    continue
                res = pair_rows(load(files[ia][sa]), load(files[ib][sb]), symmetric=ia == ib)
                for sub, m in res.items():
                    rows.append(dict(dataset=ds, k=k, implementation_a=ia, implementation_b=ib, order_a=sa,
                                     order_b=sb, subset=sub, **m))
                done.add((ds, k, ia, ib, sa, sb))
                new += 1
                print(ds, f"k{k:g}", ia, ib, sa, sb,
                      " ".join(f"{s} AMI {m['ami']:.3f}" for s, m in res.items()), flush=True)
                if new % 50 == 0:
                    flush()
            if new:
                flush()
    if rows:
        flush()
    print("wrote", OUTF)


if __name__ == "__main__":
    main()
