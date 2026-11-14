"""Shared helpers for the refinement experiment: folders, thresholds, input order, label files.

Paths come from the repo config (scripts/common/paths.py): labels and work files under BB_REFINE_POST_LABELS
(outside git), tables in data/refinement/ of the results folder, working figures under BB_REFINE_POST_LABELS/figures.
Implementation keys: serial = bblean serial, parallel = bblean split-merge (bb multiround, P = 36),
batched = bblean batched (multithreaded CPU; same partitions as nvMolKit batched).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.common import paths as P  # noqa: E402

BF = 1024
PLAYERS = ("serial", "parallel", "batched")       # bblean serial, bblean split-merge (P = 36), bblean batched
PROCS = {"rc0": 10, "rc1": 5, "ref": 5}           # procedure -> iterations run
OUT_DIR = P.path("BB_REFINE_POST_LABELS")
CSV_DIR = P.REFINEMENT
FIG_DIR = OUT_DIR / "figures"
threshold, sigma, perm = P.threshold, P.sigma, P.perm


def label_path(ds, player, proc, k, seed, it):
    """proc 'none' with it 0 = the unrefined clustering."""
    return OUT_DIR / "labels" / ds / player / proc / f"k{k:g}_s{seed}_it{it}.npy"


def save_labels(path, lab, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npy")
    np.save(tmp, lab.astype(np.int32)); tmp.rename(path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))


def load_labels(path):
    return np.load(path) if path.exists() and path.with_suffix(".json").exists() else None


def canon(lab):
    """Relabel clusters by first appearance (to compare partitions)."""
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inv]


def git_commit(path):
    import subprocess
    return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
