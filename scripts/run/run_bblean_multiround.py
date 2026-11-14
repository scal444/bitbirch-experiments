"""Quality labels from bblean split-merge (bb multiround), bblean's own parallel mode for large libraries.

  python scripts/run/run_bblean_multiround.py DS [K ...] [--seeds ...] [--files 36] [--ps 36]
         [--bblean tree|upstream]

The shuffled input (same input order as every implementation) is cut into --files equal .npy parts in a scratch
dir under BB_OUT/tmp, then run as
  bb multiround <dir> --ps P -b 1024 -t T   (bblean defaults for everything else:
  round 1 'diameter' per file, then 'full' refinement of each file's largest cluster,
  1 midsection round + final round with 'tolerance-diameter', tolerance 0.05, bin size 10)
--bblean tree (default) uses the build at BB_BBLEAN_TREE; upstream uses the bblean release installed in the
environment. Both give the same clusters (the upstream run is only a check: its label folder is
`bblean_mr<P>_upstream`, which the metrics skip).
Writes BB_OUT/labels/DS/bblean_mr<P>/kK/sSEED.npy (+ .json). Time = wall time of the bb command (process start-up
and file I/O included), plus bblean's own per-round timings.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import label_folder  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("ks", nargs="*", type=float, default=list(P.KS))
ap.add_argument("--seeds", nargs="*", type=int)
ap.add_argument("--files", type=int, default=36)
ap.add_argument("--ps", type=int, default=36)
ap.add_argument("--bblean", choices=["tree", "upstream"], default="tree")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

method = label_folder("bblean_split_merge", a.ps) + ("" if a.bblean == "tree" else "_upstream")
d = P.DATASETS[a.ds]
fps = np.load(d["fps"], mmap_mode="r")
n = len(fps)
env = dict(os.environ, BITBIRCH_BUILD_CPP="1")
if a.bblean == "tree":
    env["PYTHONPATH"] = str(P.path("BB_BBLEAN_TREE"))
else:  # the bblean 0.10.3 installed in the conda env (built with BITBIRCH_BUILD_CPP=1, see SETUP.md)
    env.pop("PYTHONPATH", None)
for k in a.ks:
    for s in a.seeds if a.seeds is not None else d["seeds"]:
        out = P.label_path(a.ds, method, k, s)
        if out.exists() and not a.force:
            continue
        t = P.threshold(a.ds, k)
        perm = P.perm(n, s)
        work = P.OUT / "tmp" / f"mr_{a.ds}_{method}_k{k:g}_s{s}"
        shutil.rmtree(work, ignore_errors=True)
        (work / "in").mkdir(parents=True)
        x = np.ascontiguousarray(fps[perm])
        for i, part in enumerate(np.array_split(x, a.files)):
            np.save(work / "in" / f"part{i:04d}.npy", part)  # bblean sorts files by name
        del x
        cmd = [sys.executable, "-c", "import sys; from bblean.cli import app; sys.argv[0]='bb'; app()",
               "multiround", str(work / "in"), "-o", str(work / "out"), "--ps", str(a.ps),
               "-b", str(P.BF), "-t", repr(t)]
        st = time.perf_counter()
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        wall = time.perf_counter() - st
        clusters = pickle.loads((work / "out" / "clusters.pkl").read_bytes())
        lab = np.full(n, -1, dtype=np.int32)
        for i, ids in enumerate(clusters):
            lab[perm[np.asarray(ids)]] = i      # ids index the shuffled, concatenated input
        assert (lab >= 0).all()
        info = dict(threshold=t, bf=P.BF, files=a.files, ps=a.ps, bblean=a.bblean, seconds=wall,
                    rounds=json.loads((work / "out" / "timings.json").read_text()),
                    max_rss=json.loads((work / "out" / "max-rss.json").read_text()),
                    clusters=int(lab.max() + 1), cmd=" ".join(cmd[3:]))
        P.save_labels(out, lab, info)
        shutil.rmtree(work)
        print(a.ds, method, f"k{k:g}", f"s{s}", {kk: v for kk, v in info.items() if kk != "cmd"}, flush=True)
