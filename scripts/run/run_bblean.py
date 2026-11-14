"""Quality labels from bblean serial or bblean batched (multithreaded CPU), untimed apart from fit().

  python scripts/run/run_bblean.py bblean_serial  DS [K ...] [--seeds ...]
  python scripts/run/run_bblean.py bblean_batched DS [K ...] [--seeds ...] [--batch-size 4096] [--threads 36]

bblean_serial imports bblean from the build at BB_BBLEAN_TREE, bblean_batched from the build at BB_BBLEAN_BATCHED
(see setup/). Branching factor 1024.
Defaults: every k in {3, 3.5, 4} and every input order (seed) of the dataset (scripts/common/paths.py).
Writes BB_OUT/labels/DS/{bblean_tree|bblean_batched}/kK/sSEED.npy (+ .json with threshold, seconds, peak memory).
Labels: cluster i = i-th entry of get_cluster_mol_ids(), mapped back to the original row order.
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import label_folder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["bblean_serial", "bblean_batched"])
ap.add_argument("ds")
ap.add_argument("ks", nargs="*", type=float, default=list(P.KS))
ap.add_argument("--seeds", nargs="*", type=int)
ap.add_argument("--batch-size", type=int, default=4096)
ap.add_argument("--threads", type=int, default=36)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

os.environ.setdefault("BITBIRCH_BUILD_CPP", "1")
sys.path.insert(0, str(P.path("BB_BBLEAN_TREE" if a.mode == "bblean_serial" else "BB_BBLEAN_BATCHED")))
import bblean  # noqa: E402
import numpy as np  # noqa: E402

method = label_folder(a.mode)
d = P.DATASETS[a.ds]
fps = np.load(d["fps"])
n = len(fps)
for k in a.ks:
    for s in a.seeds if a.seeds is not None else d["seeds"]:
        out = P.label_path(a.ds, method, k, s)
        if out.exists() and not a.force:
            continue
        perm = P.perm(n, s)
        x = np.ascontiguousarray(fps[perm])
        t = P.threshold(a.ds, k)
        secs = []
        for _ in range(a.repeats):
            kw = dict(batch_size=a.batch_size, n_threads=a.threads) if a.mode == "bblean_batched" else {}
            tree = bblean.BitBirch(threshold=t, branching_factor=P.BF, merge_criterion="diameter", **kw)
            st = time.perf_counter()
            tree.fit(x)
            secs.append(time.perf_counter() - st)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB, this process so far (incl. input)
        lab = np.empty(n, dtype=np.int32)
        for i, ids in enumerate(tree.get_cluster_mol_ids()):
            lab[perm[np.asarray(ids)]] = i
        info = dict(threshold=t, bf=P.BF, seconds=min(secs), seconds_all=secs, clusters=int(lab.max() + 1),
                    bblean=str(Path(bblean.__file__).parent.parent), peak_rss_gib=peak)
        if a.mode == "bblean_batched":
            info.update(batch_size=a.batch_size, threads=a.threads)
        P.save_labels(out, lab, info)
        print(a.ds, method, f"k{k:g}", f"s{s}", info, flush=True)
