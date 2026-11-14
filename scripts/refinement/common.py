"""Shared helpers for the refinement experiment: config, thresholds, input order, label files.

Paths come from the repo config (scripts/common/paths.py): labels and work files under BB_REFINE_POST_LABELS
(outside git), tables in data/refinement/ of the results folder, working figures under BB_REFINE_POST_LABELS/figures. Earlier labels in
BB_BP_LABELS (optional) serve as references for checks. A JSON file passed with --paths (or $REFINE_POST_PATHS)
overrides any key.
Implementation keys: serial = bblean serial, parallel = bblean split-merge (bb multiround, P = 36),
batched = bblean batched (multithreaded CPU; same partitions as nvMolKit batched).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.common import paths as P  # noqa: E402

BF = 1024
PLAYERS = ("serial", "parallel", "batched")       # bblean serial, bblean split-merge (P = 36), bblean batched
PROCS = {"rc0": 10, "rc1": 5, "ref": 5}           # procedure -> iterations run


def load_cfg(path=None):
    cfg = dict(
        out_dir=str(P.path("BB_REFINE_POST_LABELS")),
        csv_dir=str(P.REFINEMENT),
        fig_dir=str(P.path("BB_REFINE_POST_LABELS") / "figures"),
        bp_scripts=str(P.REPO / "scripts" / "metrics"),
        metrics_ext=str(P.REPO / "scripts" / "metrics"),
        thresholds_json=str(P.thresholds_json()),
        ref_labels=str(P.opt_path("BB_BP_LABELS", P.OUT / "earlier_labels")) + "/{ds}/{method}/k{k:g}/s{seed}.npy",
        actives=str(P.path("BB_ACTIVES")),
        target_sets=["chembl34", "c35"],
        datasets={ds: dict(fps=str(d["fps"]), scaf=str(d["scaf"])) for ds, d in P.DATASETS.items()},
    )
    p = path or os.environ.get("REFINE_POST_PATHS")
    if p:
        cfg.update(json.loads(Path(p).read_text()))
    cfg["_T"] = json.loads(Path(cfg["thresholds_json"]).read_text())
    return cfg


def isim(cfg, ds): return cfg["_T"][ds]["isim"]
def sigma(cfg, ds): return cfg["_T"][ds]["sigma_n50"]
def threshold(cfg, ds, k): return isim(cfg, ds) + k * sigma(cfg, ds)
def perm(n, seed): return np.random.default_rng(seed).permutation(n)


def label_path(cfg, ds, player, proc, k, seed, it):
    """proc 'none' with it 0 = the unrefined clustering."""
    return Path(cfg["out_dir"]) / "labels" / ds / player / proc / f"k{k:g}_s{seed}_it{it}.npy"


def ref_label_path(cfg, ds, method, k, seed):
    return Path(cfg["ref_labels"].format(ds=ds, method=method, k=k, seed=seed))


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
