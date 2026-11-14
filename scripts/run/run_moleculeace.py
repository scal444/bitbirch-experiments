"""MoleculeACE runs: the 30 MoleculeACE sets (RDKit 2048 bits) at fixed thresholds, as in the BitBIRCH paper (2025).

  python scripts/run/run_moleculeace.py PLAYER [PLAYER ...] [--sets CHEMBL262_Ki ...] [--bf 50]

PLAYER:
  bblean_tree     bblean serial (build BB_BBLEAN_TREE)
  bblean_batched  bblean batched, multithreaded CPU (build BB_BBLEAN_BATCHED, 36 threads)
  bblean_mr10     bblean split-merge (bb multiround) with 10 files / 10 processes, as the paper's "10 initial splits"
  nvmolkit_main   nvMolKit batched (GPU, batch 4096; BB_NVMOLKIT_PY)
  butina          nvMolKit fused Butina (GPU), distance cutoff 1 - t
bblean_tree and bblean_batched import different builds, so run them in separate calls.
Thresholds 0.30-0.80 step 0.05; branching factor 50 (the paper's setting); diameter merge; input order seed 0.
Input: BB_DATA/moleculeace/<set>/rdkit_2048.npy. Writes BB_OUT/moleculeace/labels/<set>/<player>.npz (key 't0.30' ... ->
int32 label per original row) and BB_OUT/moleculeace/runs_<player>.csv (set, n, threshold, seconds, clusters).
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

import numpy as np  # noqa: E402

MACE = P.DATA / "moleculeace"
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]
ap = argparse.ArgumentParser()
ap.add_argument("players", nargs="+", choices=["bblean_tree", "bblean_mr10", "nvmolkit_main", "butina",
                                                                "bblean_batched"])
ap.add_argument("--sets", nargs="*")
ap.add_argument("--bf", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sets = a.sets or sorted(p.name for p in MACE.glob("CHEMBL*") if (p / "rdkit_2048.npy").exists())
out_root = P.OUT / "moleculeace"
os.environ.setdefault("BITBIRCH_BUILD_CPP", "1")
if "bblean_tree" in a.players and "bblean_batched" in a.players:
    sys.exit("bblean_tree and bblean_batched import different bblean builds: run them in separate calls")
if "bblean_tree" in a.players:
    sys.path.insert(0, str(P.path("BB_BBLEAN_TREE")))
    import bblean  # noqa: E402
if "bblean_batched" in a.players:  # bblean batched (multithreaded CPU)
    sys.path.insert(0, str(P.path("BB_BBLEAN_BATCHED")))
    import bblean  # noqa: E402
if {"nvmolkit_main", "butina"} & set(a.players):
    sys.path.insert(0, str(P.path("BB_NVMOLKIT_PY")))
    import torch  # noqa: E402
    from nvmolkit.clustering import bitbirch, fused_butina  # noqa: E402


def run_one(player, x, t, work):
    """x: packed uint8 fingerprints in the shuffled order. Returns labels in that order."""
    n = len(x)
    if player == "bblean_tree":
        tree = bblean.BitBirch(threshold=t, branching_factor=a.bf, merge_criterion="diameter")
        tree.fit(x)
        groups = tree.get_cluster_mol_ids()
    elif player == "bblean_batched":
        tree = bblean.BitBirch(threshold=t, branching_factor=a.bf, merge_criterion="diameter", batch_size=4096,
                               n_threads=36)
        tree.fit(x)
        groups = tree.get_cluster_mol_ids()
    elif player == "bblean_mr10":
        shutil.rmtree(work, ignore_errors=True)
        (work / "in").mkdir(parents=True)
        for i, part in enumerate(np.array_split(x, 10)):
            np.save(work / "in" / f"part{i:04d}.npy", part)
        env = dict(os.environ, PYTHONPATH=str(P.path("BB_BBLEAN_TREE")))
        cmd = [sys.executable, "-c", "import sys; from bblean.cli import app; sys.argv[0]='bb'; app()",
               "multiround", str(work / "in"), "-o", str(work / "out"), "--ps", "10", "-b", str(a.bf), "-t", repr(t)]
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        groups = pickle.loads((work / "out" / "clusters.pkl").read_bytes())
        shutil.rmtree(work)
    else:
        w = torch.from_numpy(np.ascontiguousarray(x).view(np.int32))
        xg = w.cuda()
        if player == "nvmolkit_main":
            lab = bitbirch(xg, t, branching_factor=a.bf, batch_size=4096).numpy()
        else:
            lab = fused_butina(xg, 1.0 - t).cluster_ids.numpy()
        del xg
        return lab.astype(np.int32)
    lab = np.full(n, -1, np.int32)
    for i, ids in enumerate(groups):
        lab[np.asarray(ids)] = i
    assert (lab >= 0).all()
    return lab


for player in a.players:
    runs_csv = out_root / f"runs_{player}.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    new = not runs_csv.exists()
    with open(runs_csv, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["set", "method", "n", "bf", "seed", "threshold", "seconds", "clusters"])
        for s in sets:
            npz = out_root / "labels" / s / f"{player}.npz"
            if npz.exists():
                continue
            fps = np.load(MACE / s / "rdkit_2048.npy")
            perm = P.perm(len(fps), a.seed)
            x = np.ascontiguousarray(fps[perm])
            res = {}
            for t in THRESHOLDS:
                st = time.perf_counter()
                lab_s = run_one(player, x, t, P.OUT / "tmp" / f"moleculeace_{player}_{s}")
                sec = time.perf_counter() - st
                lab = np.empty(len(fps), np.int32)
                lab[perm] = lab_s
                res[f"t{t:.2f}"] = lab
                wr.writerow([s, player, len(fps), a.bf, a.seed, f"{t:.2f}", f"{sec:.4f}", int(lab.max() + 1)])
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz.with_suffix(".partial.npz"), **res)
            npz.with_suffix(".partial.npz").rename(npz)
            fh.flush()
            print(player, s, len(fps), [int(v.max() + 1) for v in res.values()], flush=True)
