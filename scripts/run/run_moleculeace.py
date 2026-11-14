"""MoleculeACE runs: the 30 MoleculeACE sets (RDKit 2048 bits) at fixed thresholds, as in the BitBIRCH paper (2025).

  python scripts/run/run_moleculeace.py IMPLEMENTATION [IMPLEMENTATION ...] [--sets CHEMBL262_Ki ...] [--bf 50]

IMPLEMENTATION:
  bblean_serial       build BB_BBLEAN_TREE
  bblean_batched      multithreaded CPU (build BB_BBLEAN_BATCHED, 36 threads)
  bblean_split_merge  bb multiround with 10 files / 10 processes, as the paper's "10 initial splits"
  nvmolkit_batched    GPU, batch 4096 (BB_NVMOLKIT_PY)
bblean_serial and bblean_batched import different builds, so run them in separate calls.
Thresholds 0.30-0.80 step 0.05; branching factor 50 (the paper's setting); diameter merge; input order seed 0.
Input: BB_DATA/moleculeace/<set>/rdkit_2048.npy. Writes BB_OUT/moleculeace/labels/<set>/<folder>.npz (key 't0.30' ...
-> int32 label per original row; folder = the implementation's label folder name, split-merge `bblean_mr10`) and
BB_OUT/moleculeace/runs_<folder>.csv (set, implementation, processes, n, bf, seed, threshold, seconds, clusters).
Untimed quality runs (seconds are indicative).
"""
import argparse
import csv
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import IMPLEMENTATIONS, label_folder  # noqa: E402

import numpy as np  # noqa: E402

MACE = P.DATA / "moleculeace"
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]
MR_P = 10  # split-merge files and processes (the paper's 10 initial splits)
ap = argparse.ArgumentParser()
ap.add_argument("implementations", nargs="+", choices=IMPLEMENTATIONS)
ap.add_argument("--sets", nargs="*")
ap.add_argument("--bf", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sets = a.sets or sorted(p.name for p in MACE.glob("CHEMBL*") if (p / "rdkit_2048.npy").exists())
out_root = P.OUT / "moleculeace"
os.environ.setdefault("BITBIRCH_BUILD_CPP", "1")
if "bblean_serial" in a.implementations and "bblean_batched" in a.implementations:
    sys.exit("bblean_serial and bblean_batched import different bblean builds: run them in separate calls")
if "bblean_serial" in a.implementations:
    sys.path.insert(0, str(P.path("BB_BBLEAN_TREE")))
    import bblean  # noqa: E402
if "bblean_batched" in a.implementations:  # multithreaded CPU
    sys.path.insert(0, str(P.path("BB_BBLEAN_BATCHED")))
    import bblean  # noqa: E402
if "nvmolkit_batched" in a.implementations:
    sys.path.insert(0, str(P.path("BB_NVMOLKIT_PY")))
    import torch  # noqa: E402
    from nvmolkit.clustering import bitbirch  # noqa: E402


def run_one(impl, x, t, work):
    """x: packed uint8 fingerprints in the shuffled order. Returns labels in that order."""
    n = len(x)
    if impl == "bblean_serial":
        tree = bblean.BitBirch(threshold=t, branching_factor=a.bf, merge_criterion="diameter")
        tree.fit(x)
        groups = tree.get_cluster_mol_ids()
    elif impl == "bblean_batched":
        tree = bblean.BitBirch(threshold=t, branching_factor=a.bf, merge_criterion="diameter", batch_size=4096,
                               n_threads=36)
        tree.fit(x)
        groups = tree.get_cluster_mol_ids()
    elif impl == "bblean_split_merge":
        shutil.rmtree(work, ignore_errors=True)
        (work / "in").mkdir(parents=True)
        for i, part in enumerate(np.array_split(x, MR_P)):
            np.save(work / "in" / f"part{i:04d}.npy", part)
        env = dict(os.environ, PYTHONPATH=str(P.path("BB_BBLEAN_TREE")))
        cmd = [sys.executable, "-c", "import sys; from bblean.cli import app; sys.argv[0]='bb'; app()",
               "multiround", str(work / "in"), "-o", str(work / "out"), "--ps", str(MR_P), "-b", str(a.bf),
               "-t", repr(t)]
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        groups = pickle.loads((work / "out" / "clusters.pkl").read_bytes())
        shutil.rmtree(work)
    else:  # nvmolkit_batched
        xg = torch.from_numpy(np.ascontiguousarray(x).view(np.int32)).cuda()
        lab = bitbirch(xg, t, branching_factor=a.bf, batch_size=4096).numpy()
        del xg
        return lab.astype(np.int32)
    lab = np.full(n, -1, np.int32)
    for i, ids in enumerate(groups):
        lab[np.asarray(ids)] = i
    assert (lab >= 0).all()
    return lab


for impl in a.implementations:
    folder = label_folder(impl, MR_P)
    procs = MR_P if impl == "bblean_split_merge" else ""
    runs_csv = out_root / f"runs_{folder}.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    new = not runs_csv.exists()
    with open(runs_csv, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["set", "implementation", "processes", "n", "bf", "seed", "threshold", "seconds", "clusters"])
        for s in sets:
            npz = out_root / "labels" / s / f"{folder}.npz"
            if npz.exists():
                continue
            fps = np.load(MACE / s / "rdkit_2048.npy")
            perm = P.perm(len(fps), a.seed)
            x = np.ascontiguousarray(fps[perm])
            res = {}
            for t in THRESHOLDS:
                st = time.perf_counter()
                lab_s = run_one(impl, x, t, P.OUT / "tmp" / f"moleculeace_{folder}_{s}")
                sec = time.perf_counter() - st
                lab = np.empty(len(fps), np.int32)
                lab[perm] = lab_s
                res[f"t{t:.2f}"] = lab
                wr.writerow([s, impl, procs, len(fps), a.bf, a.seed, f"{t:.2f}", f"{sec:.4f}", int(lab.max() + 1)])
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz.with_suffix(".partial.npz"), **res)
            npz.with_suffix(".partial.npz").rename(npz)
            fh.flush()
            print(impl, s, len(fps), [int(v.max() + 1) for v in res.values()], flush=True)
