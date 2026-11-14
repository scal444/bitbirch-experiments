"""Quality and agreement tables for the refinement experiment's labels.

   python scripts/refinement/analyze.py DS [DS ...] [--k 3.5] [--seeds 0] [--paths cfg.json]   (GPU)

Quality: scripts/metrics/metrics.py `evaluate` (clusters, singletons, size-weighted iSIM, scaffold purity, medoid
iSIM, ARI/AMI vs Murcko scaffolds and ChEMBL targets) on every saved labelling -> data/refinement/quality_<ds>.csv.
Agreement (GPU AMI from ext_metrics, pair counts on CPU) -> data/refinement/agreement_<ds>.csv, kinds:
  players  - two implementations, same procedure and iteration (it0 = unrefined)
  input    - a refined labelling vs the same implementation's unrefined labelling
  previous - iteration i vs i-1 of the same implementation and procedure
  seeds    - one implementation, procedure and iteration, two input orders (only when several seeds are given)
Pair columns: rand = share of all molecule pairs treated the same way (together in both or apart in both);
pair_jaccard = of the pairs put together by either labelling, the share put together by both.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def import_metrics(cfg):
    sys.path.insert(0, cfg["bp_scripts"]); sys.path.insert(0, cfg["metrics_ext"])
    import metrics as M
    import registry as R
    for ds, d in cfg["datasets"].items():
        R.DATASETS.setdefault(ds, {}).update(fps=Path(d["fps"]), scaf=Path(d["scaf"]))
    M.ACTIVES = Path(cfg["actives"]); M.TARGET_SETS = set(cfg["target_sets"])
    return M


def pairs(sizes):
    s = sizes.astype(np.int64)
    return int((s * (s - 1) // 2).sum())


def pair_stats(a, b):
    n = len(a)
    ab = np.unique(a.astype(np.int64) * (int(b.max()) + 1) + b, return_counts=True)[1]
    pa, pb, pab = pairs(np.bincount(a)), pairs(np.bincount(b)), pairs(ab)
    tot = n * (n - 1) // 2
    return dict(rand=1 - (pa + pb - 2 * pab) / tot, pair_jaccard=pab / (pa + pb - pab) if pa + pb - pab else 1.0,
                pairs_a=pa, pairs_b=pb, pairs_both=pab)


def agree(M, a, b):
    e = M.external_metrics(a, b)
    return dict(ari=float(e["ari"]), ami=float(e["ami"]), identical=bool(np.array_equal(C.canon(a), C.canon(b))),
                **pair_stats(a, b))


def run(cfg, M, ds, k, seed, quality=True):
    labs = {}
    for player in C.PLAYERS:
        l0 = C.load_labels(C.label_path(cfg, ds, player, "none", k, seed, 0))
        if l0 is None:
            continue
        for proc, its in C.PROCS.items():
            labs[(player, proc, 0)] = l0
            for it in range(1, its + 1):
                l = C.load_labels(C.label_path(cfg, ds, player, proc, k, seed, it))
                if l is not None:
                    labs[(player, proc, it)] = l
    players = sorted({p for p, _, _ in labs}, key=C.PLAYERS.index)
    ident = dict(dataset=ds, k=k, seed=seed)

    ctx = M.load_context(ds) if quality else None
    qrows, done = [], {}
    for (player, proc, it), lab in (labs.items() if quality else []):
        key = (player, "none", 0) if it == 0 else (player, proc, it)
        if key not in done:
            row, _ = M.evaluate(ctx, lab, {"threshold": None}, None)
            done[key] = {c: v for c, v in row.items() if not c.endswith("seconds")}
        qrows.append({**ident, "player": player, "proc": proc, "it": it, **done[key]})
    del ctx
    q = pd.DataFrame(qrows)

    arows, cache = [], {}
    def ag(x, y):
        kx = tuple(sorted([x, y]))
        if kx not in cache:
            cache[kx] = agree(M, labs[x], labs[y])
        return cache[kx]
    for (player, proc, it) in labs:
        if it > 0:
            arows.append({**ident, "kind": "input", "player_a": player, "player_b": player, "proc": proc, "it": it,
                          **ag((player, proc, it), (player, proc, 0))})
            arows.append({**ident, "kind": "previous", "player_a": player, "player_b": player, "proc": proc,
                          "it": it, **ag((player, proc, it), (player, proc, it - 1))})
    for pa, pb in combinations(players, 2):
        for proc, its in C.PROCS.items():
            for it in range(its + 1):
                if (pa, proc, it) in labs and (pb, proc, it) in labs:
                    arows.append({**ident, "kind": "players", "player_a": pa, "player_b": pb, "proc": proc, "it": it,
                                  **ag((pa, proc, it), (pb, proc, it))})
    a = pd.DataFrame(arows)

    # cross-check (optional): serial + recluster x5 must equal earlier bblean_rc0 / bblean_rc1 labels in BB_BP_LABELS
    for proc, meth in (("rc0", "bblean_rc0"), ("rc1", "bblean_rc1")):
        p = C.ref_label_path(cfg, ds, meth, k, seed)
        if ("serial", proc, 5) in labs and p.exists():
            same = np.array_equal(C.canon(labs[("serial", proc, 5)]), C.canon(np.load(p)))
            print(f"{ds} k{k:g} s{seed}: serial {proc} it5 equals bp {meth}: {same}", flush=True)
    return q, a, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+"); ap.add_argument("--k", type=float, default=3.5)
    ap.add_argument("--seeds", default="0"); ap.add_argument("--paths")
    ap.add_argument("--quality-seeds", default="0", help="seeds that get the quality metrics (others: agreement only)")
    a = ap.parse_args()
    cfg = C.load_cfg(a.paths); M = import_metrics(cfg)
    import torch
    torch.set_num_threads(8)
    out = Path(cfg["csv_dir"]); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        qs, as_, by_seed = [], [], {}
        for seed in map(int, a.seeds.split(",")):
            q, ag, by_seed[seed] = run(cfg, M, ds, a.k, seed, str(seed) in a.quality_seeds.split(",")); qs.append(q); as_.append(ag)
        rows = []
        for sa, sb in combinations(sorted(by_seed), 2):
            for key in by_seed[sa]:
                if key in by_seed[sb]:
                    player, proc, it = key
                    rows.append(dict(dataset=ds, k=a.k, seed=sa, seed_b=sb, kind="seeds", player_a=player,
                                     player_b=player, proc=proc, it=it, **agree(M, by_seed[sa][key], by_seed[sb][key])))
        as_.append(pd.DataFrame(rows))
        del by_seed
        pd.concat(qs).to_csv(out / f"quality_{ds}.csv", index=False)
        pd.concat(as_).to_csv(out / f"agreement_{ds}.csv", index=False)
        print(f"{ds}: wrote data/refinement/quality_{ds}.csv, data/refinement/agreement_{ds}.csv", flush=True)


if __name__ == "__main__":
    main()
