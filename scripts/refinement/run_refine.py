"""Apply bblean's refinement procedures to each implementation's BitBIRCH tree and save labels per iteration.

   python scripts/refinement/run_refine.py DS [--k 3.5] [--seed 0] [--players serial,batched,parallel]
          [--threads 8]

bblean is imported from one build per process, so serial and parallel run in one call and batched in another:
  serial, parallel -> BB_BBLEAN_TREE
  batched          -> BB_BBLEAN_BATCHED
Implementations (threshold iSIM + k*sigma, bf 1024, diameter):
  serial   - bblean serial, BitBirch.fit
  batched  - bblean batched, BitBirch(batch_size=4096, n_threads=T): nvMolKit's batched algorithm in a real bblean
             tree (same partition as nvMolKit batched on the GPU)
  parallel - bblean split-merge, run_multiround_bitbirch = `bb multiround <dir> --ps 36 -b 1024 -t T` (input cut
             into 36 equal parts, 36 processes, bblean defaults otherwise: full initial refinement, 1 midsection
             round, bin 10, tolerance-diameter mid/final rounds), save_tree=True. Same command as
             scripts/run/run_bblean_multiround.py.
The unrefined serial / batched labels are checked against this repo's labels (`bblean_tree`, `nvmolkit_main`) when
present.
Procedures (bblean 0.10.3 methods only), one call per iteration:
  rc0 - recluster_inplace(iterations=1, extra_threshold=0)       merge 'diameter' at t
  rc1 - recluster_inplace(iterations=1, extra_threshold=sigma)   merge 'diameter', threshold grows by sigma each iteration
  ref - set_merge('tolerance-diameter', tolerance=0.05, threshold=t); refine_inplace(X, n_largest=1)
        (= `bb run --refine-num 1 --refine-rounds R`, and multiround's initial 'full' refinement step)
Output: BB_REFINE_POST_LABELS/labels/<ds>/<player>/<proc>/k<k>_s<seed>_it<i>.npy (original row order), it0 under
proc 'none'.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BITBIRCH_BUILD_CPP", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def labels_of(tree, perm):
    lab = np.empty(len(perm), dtype=np.int32)
    for i, ids in enumerate(tree.get_cluster_mol_ids()):
        lab[perm[np.asarray(ids)]] = i  # ids index the permuted array -> original rows
    return lab


def info_of(lab, tree, **kw):
    sz = np.bincount(lab)
    return dict(clusters=int(len(sz)), singletons=int((sz == 1).sum()), largest=int(sz.max()),
                threshold=float(tree.threshold), **kw)


MR_PROCS = 36  # processes (= input parts) for the parallel player


def use_build(players):
    """Put the bblean build for these players first on sys.path (and PYTHONPATH, for multiround's worker processes)."""
    keys = {"BB_BBLEAN_BATCHED" if p == "batched" else "BB_BBLEAN_TREE" for p in players}
    assert len(keys) == 1, f"run batched in its own call (builds {keys})"
    root = C.P.path(keys.pop())
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root)
    import bblean
    assert Path(bblean.__file__).parent.parent == root, bblean.__file__
    return root


class Players:
    def __init__(self, ds, k, seed, threads, work):
        import bblean
        self.bb, self.ds, self.k, self.seed, self.threads = bblean, ds, k, seed, threads
        fps = np.load(C.P.DATASETS[ds]["fps"])
        self.perm = C.perm(len(fps), seed)
        self.X = np.ascontiguousarray(fps[self.perm]); del fps
        self.t = C.threshold(ds, k)
        self.work = work / ds / f"k{k:g}_s{seed}"

    def build(self, player):
        bb, t = self.bb, self.t
        if player == "serial":
            tr = bb.BitBirch(threshold=t, branching_factor=C.BF, merge_criterion="diameter"); tr.fit(self.X)
        elif player == "batched":
            tr = bb.BitBirch(threshold=t, branching_factor=C.BF, merge_criterion="diameter", batch_size=4096,
                             n_threads=self.threads)
            tr.fit(self.X)
        elif player == "parallel":
            pkl = self.work / "multiround" / "bitbirch.pkl"
            if not pkl.exists():
                self.run_multiround()
            tr = bb.BitBirch.load(pkl)
        return tr

    def run_multiround(self):
        from bblean.multiround import run_multiround_bitbirch
        parts = self.work / "parts"; out = self.work / "multiround"
        parts.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
        files = []
        for i, chunk in enumerate(np.array_split(self.X, MR_PROCS)):  # 36 equal parts, as run_bblean_multiround.py
            f = parts / f"part{i:04d}.npy"; np.save(f, chunk); files.append(f)
        run_multiround_bitbirch(files, out, num_initial_processes=MR_PROCS, branching_factor=C.BF, threshold=self.t,
                                initial_merge_criterion="diameter", save_tree=True)
        for f in files:
            f.unlink()


def step(tree, proc, it, P):
    if proc in ("rc0", "rc1"):
        if it == 1:
            tree.set_merge("diameter", threshold=P.t)  # parallel's saved tree ends in tolerance-diameter
        tree.recluster_inplace(iterations=1, extra_threshold=0.0 if proc == "rc0" else C.sigma(P.ds))
    elif proc == "ref":
        if it == 1:
            tree.set_merge("tolerance-diameter", tolerance=0.05, threshold=P.t)
        tree.refine_inplace(P.X, n_largest=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("--k", type=float, default=3.5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--players", default="serial,batched"); ap.add_argument("--procs", default="rc0,rc1,ref")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    root = use_build(a.players.split(","))
    build = dict(bblean=str(root), commit=C.git_commit(root))
    P = Players(a.ds, a.k, a.seed, a.threads, C.OUT_DIR / "work")
    for player in a.players.split(","):
        procs = [p for p in a.procs.split(",")
                 if not C.label_path(a.ds, player, p, a.k, a.seed, C.PROCS[p]).exists()]
        p0 = C.label_path(a.ds, player, "none", a.k, a.seed, 0)
        if not procs and p0.exists():
            continue
        s = time.perf_counter(); tree = P.build(player); fit_s = time.perf_counter() - s
        lab = labels_of(tree, P.perm)
        check = {}
        ref = {"serial": "bblean_tree", "batched": "nvmolkit_main"}.get(player)
        if ref and (rp := C.P.label_path(a.ds, ref, a.k, a.seed)).exists():
            check = {f"equals_{ref}": bool(np.array_equal(C.canon(lab), C.canon(np.load(rp))))}
        C.save_labels(p0, lab, info_of(lab, tree, fit_seconds=fit_s, **check, **build))
        print(f"{a.ds} {player}: {lab.max() + 1} clusters, fit {fit_s:.1f}s {check}", flush=True)
        for proc in procs:
            if tree is None:
                s = time.perf_counter(); tree = P.build(player); fit_s = time.perf_counter() - s
            for it in range(1, C.PROCS[proc] + 1):
                s = time.perf_counter(); step(tree, proc, it, P); sec = time.perf_counter() - s
                lab = labels_of(tree, P.perm)
                C.save_labels(C.label_path(a.ds, player, proc, a.k, a.seed, it), lab,
                              info_of(lab, tree, step_seconds=sec, **build, merge=type(tree._merge_accept_fn).__name__
                                      if hasattr(tree, "_merge_accept_fn") else None))
                print(f"{a.ds} {player} {proc} it{it}: {lab.max() + 1} clusters, {int((np.bincount(lab) == 1).sum())}"
                      f" singletons, t {tree.threshold:.4f}, {sec:.1f}s", flush=True)
            tree = None  # fresh tree for the next procedure (deterministic rebuild)


if __name__ == "__main__":
    main()
