"""Quality metrics for every labelling under BB_OUT/labels.

   python scripts/metrics/metrics.py [DS ...] [--force]     (output data/quality/by_run/ of the results folder)

For every labels/<ds>/<method>/k<k>/s<seed>.npy that has its .json, writes one row to metrics/<ds>.csv and one
row per k_top to metrics/topk_<ds>.csv: clusters, singletons, within-cluster iSIM, scaffold purity, medoid iSIM,
and ARI / AMI / V-measure against Murcko scaffolds, ChEMBL targets (BB_ACTIVES; chembl34 and c35) and the Butina
reference (`butina_matched` labels, only where an earlier run made them; the columns stay empty otherwise). Incremental: a labelling is skipped when its row exists,
its label file is unchanged (mtime) and its Butina column state is current (recomputed once butina_matched appears).

GPU (torch) work: per-cluster bit sums, iSIM, medoids, external metrics. Scaffold / target majority counts run in CPU
threads at the same time.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R  # noqa: E402

from ext_metrics import external_metrics  # noqa: E402

KTOPS = (5, 20, 100, 1000)
ACTIVES = R.ACTIVES
TARGET_SETS = {"chembl34", "c35"}
EXT = ("ari", "ami", "v_measure", "homogeneity", "completeness")
REFS = ("scaf", "tgt", "butina")
CHUNK = 1 << 16  # molecules per GPU chunk (whole clusters; a bigger cluster gets its own chunk)
OUT = R._P.QUALITY / "by_run"  # small per-run CSVs, kept in the repository
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- iSIM / medoids (GPU)

def isim_from_sums(kq, sq, n):
    """bblean jt_isim_from_sum, vectorised. kq = sum of column sums, sq = sum of squared column sums (int64), n = rows.

    Same operation order as bblean's C++ (a = (sq - kq) / 2.0; a / ((a + n*kq) - sq)); 1 when no bit is set."""
    a = (sq - kq).double() / 2.0
    val = a / ((a + (n * kq).double()) - sq.double())
    return torch.where(kq == 0, torch.ones_like(val), val)


def unpack(packed, shifts):
    return ((packed.unsqueeze(-1) >> shifts) & 1).reshape(packed.shape[0], -1)


def column_sums(fps, rows, shifts):
    """Column sums (int64) of the fingerprints `rows`, in pieces of CHUNK rows."""
    acc = torch.zeros(fps.shape[1] * 8, dtype=torch.int64, device=fps.device)
    for b in range(0, len(rows), CHUNK):
        acc += unpack(fps[rows[b:b + CHUNK]], shifts).sum(0, dtype=torch.int64)
    return acc


def isim_of_rows(fps, rows, shifts):
    if len(rows) < 2:
        return float("nan")
    s = column_sums(fps, rows, shifts)
    return float(isim_from_sums(s.sum(), (s * s).sum(), len(rows)))


def cluster_pass(fps, codes_t, sizes_np):
    """Per-cluster iSIM and medoid row, plus library column sums.

    codes_t: int64 cluster code per molecule (0..C-1) on the device. Medoid = bblean jt_isim_medoid on the cluster's
    members in ascending row order: first member for clusters of 1-2, else argmin of the complementary iSIM (iSIM of
    the cluster without that member), first on ties."""
    dev = fps.device
    shifts = torch.arange(8, device=dev, dtype=torch.uint8)
    C = len(sizes_np)
    order = torch.argsort(codes_t, stable=True)  # rows grouped by cluster, ascending row inside each
    sizes = torch.from_numpy(sizes_np).to(dev)
    starts_np = np.concatenate([[0], np.cumsum(sizes_np)[:-1]])
    grp = starts_np // CHUNK
    cuts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1], True])
    isim = torch.full((C,), float("nan"), dtype=torch.float64, device=dev)
    medoid = torch.empty(C, dtype=torch.int64, device=dev)
    lib = torch.zeros(fps.shape[1] * 8, dtype=torch.int64, device=dev)
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        r0 = int(starts_np[c0]); r1 = int(starts_np[c1 - 1] + sizes_np[c1 - 1])
        rows = order[r0:r1]
        sz = sizes[c0:c1]
        local = torch.repeat_interleave(torch.arange(c1 - c0, device=dev), sz)
        bits = unpack(fps[rows], shifts)
        counts = torch.zeros((c1 - c0, bits.shape[1]), dtype=torch.int32, device=dev)
        counts.index_add_(0, local, bits.to(torch.int32))
        lib += counts.sum(0, dtype=torch.int64)
        kq = counts.sum(1, dtype=torch.int64)
        sq = (counts.to(torch.int64) ** 2).sum(1)
        isim[c0:c1] = torch.where(sz >= 2, isim_from_sums(kq, sq, sz), isim[c0:c1])
        # complementary iSIM of every member: column sums minus the member, n - 1 rows
        pop = bits.sum(1, dtype=torch.int64)
        dot = (bits.to(torch.int32) * counts[local]).sum(1, dtype=torch.int64)
        del bits
        kq_i = kq[local] - pop
        sq_i = sq[local] - 2 * dot + pop
        comp = isim_from_sums(kq_i, sq_i, sz[local] - 1)
        comp = torch.where(sz[local] >= 3, comp, torch.zeros_like(comp))  # n < 3: all tie -> first member
        best = torch.full((c1 - c0,), float("inf"), dtype=torch.float64, device=dev)
        best = best.scatter_reduce(0, local, comp, "amin")
        pos = torch.arange(r1 - r0, device=dev)
        cand = torch.where(comp == best[local], pos, torch.full_like(pos, r1 - r0))
        first = torch.full((c1 - c0,), r1 - r0, dtype=torch.int64, device=dev).scatter_reduce(0, local, cand, "amin")
        medoid[c0:c1] = rows[first]
    return isim, medoid, lib


# ---------------------------------------------------------------- majority labels (CPU)

def modal(codes, ref, C):
    """Per cluster: most common ref value (smallest id on ties), its count and the number of members with ref >= 0."""
    ok = ref >= 0
    cc, rr = codes[ok].astype(np.int64), ref[ok].astype(np.int64)
    width = int(rr.max()) + 1 if len(rr) else 1
    key, cnt = np.unique(cc * width + rr, return_counts=True)
    kc = key // width
    mode = np.full(C, -1, np.int64); mcount = np.zeros(C, np.int64)
    valid = np.bincount(cc, minlength=C)
    if len(key):
        st = np.flatnonzero(np.r_[True, kc[1:] != kc[:-1]])
        mx = np.maximum.reduceat(cnt, st)
        seg = np.repeat(np.arange(len(st)), np.diff(np.r_[st, len(key)]))
        hit = np.flatnonzero(cnt == mx[seg])
        _, firsts = np.unique(seg[hit], return_index=True)  # first (= smallest ref id) maximum per cluster
        sel = hit[firsts]
        mode[kc[sel]] = key[sel] % width
        mcount[kc[sel]] = cnt[sel]
    return mode, mcount, valid


# ---------------------------------------------------------------- per-dataset references

def load_targets(ds, n):
    """Target code per molecule for molecules with exactly one annotated ChEMBL target, else -1."""
    ids = (Path(R.DATASETS[ds]["fps"]).parent / "ids.txt").read_text().split()
    assert len(ids) == n, (ds, len(ids), n)
    a = pd.read_csv(ACTIVES, sep="\t").drop_duplicates()
    nt = a.groupby("chembl_id")["target_chembl_id"].nunique()
    one = a[a["chembl_id"].isin(nt.index[nt == 1])].drop_duplicates("chembl_id")
    tcode = {t: i for i, t in enumerate(sorted(one["target_chembl_id"].unique()))}
    m = dict(zip(one["chembl_id"], one["target_chembl_id"].map(tcode)))
    return np.fromiter((m.get(i, -1) for i in ids), np.int64, n)


def load_labels(p):
    j = p.with_suffix(".json")
    if not j.exists():
        return None, None
    try:
        info = json.loads(j.read_text())
        lab = np.load(p)
    except (ValueError, OSError):
        return None, None  # still being written
    return lab, info


def codes_of(lab):
    """Labels -> codes 0..C-1 keeping the label-id order (used for size ties)."""
    lab = np.asarray(lab).astype(np.int64, copy=False)
    if lab.min() == 0:
        sz = np.bincount(lab)
        if sz.all():
            return lab, sz
    _, inv = np.unique(lab, return_inverse=True)
    return inv, np.bincount(inv)


def ext_cols(prefix, m):
    return {f"{prefix}_{k}": m[k] for k in EXT} if m is not None else {f"{prefix}_{k}": np.nan for k in EXT}


# ---------------------------------------------------------------- one labelling

def evaluate(ctx, lab, info, butina):
    t0 = time.perf_counter()
    n = len(lab)
    codes, sizes = codes_of(lab)
    C = len(sizes)
    scaf, tgt = ctx["scaf"], ctx["tgt"]
    multi = sizes >= 2
    rank = np.lexsort((np.arange(C), -sizes))  # largest first, ties by label id
    pool = ThreadPoolExecutor(2)
    f_scaf = pool.submit(modal, codes, scaf, C)
    f_tgt = pool.submit(modal, codes, tgt, C) if tgt is not None else None

    fps = ctx["fps_gpu"]
    codes_t = torch.from_numpy(codes).to(DEV)
    isim_t, medoid_t, lib = cluster_pass(fps, codes_t, sizes)
    shifts = torch.arange(8, device=DEV, dtype=torch.uint8)
    mult_t = torch.from_numpy(multi).to(DEV)
    med_all = isim_of_rows(fps, medoid_t, shifts)
    med_multi = isim_of_rows(fps, medoid_t[mult_t], shifts)
    lib_isim = float(isim_from_sums(lib.sum(), (lib * lib).sum(), n))
    cl_isim = isim_t.cpu().numpy()
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    t_gpu = time.perf_counter() - t0
    ext = {"scaf": external_metrics(codes, scaf)}
    if tgt is not None:
        ext["tgt"] = external_metrics(codes, tgt, subset=tgt >= 0)
    if butina is not None:
        ext["butina"] = external_metrics(codes, butina)
    top_members, ext_top = {}, {}
    for kt in KTOPS:
        sel = np.zeros(C, bool); sel[rank[:kt]] = True
        mem = sel[codes]
        top_members[kt] = (sel, mem)
        e = {"scaf": external_metrics(codes, scaf, subset=mem)}
        if tgt is not None:
            e["tgt"] = external_metrics(codes, tgt, subset=mem & (tgt >= 0))
        if butina is not None:
            e["butina"] = external_metrics(codes, butina, subset=mem)
        ext_top[kt] = e
    t_ext = time.perf_counter() - t0 - t_gpu

    s_mode, s_cnt, s_valid = f_scaf.result()
    t_mode, t_cnt, t_valid = f_tgt.result() if f_tgt else (None, None, None)
    pool.shutdown()

    def wmean(v, w):
        return float((v * w).sum() / w.sum()) if w.sum() > 0 else np.nan

    def purity(mask):
        m = mask & multi & (s_valid > 0)
        return float(s_cnt[m].sum() / s_valid[m].sum()) if m.any() else np.nan

    singles = int((sizes == 1).sum())
    row = dict(n_molecules=n, threshold=info.get("threshold"), seconds=info.get("seconds"), clusters=C,
               singletons=singles, singleton_fraction=singles / n, singletons_per_cluster=singles / C,
               largest_cluster=int(sizes.max()), nonsingleton_clusters=int(multi.sum()),
               isim=wmean(cl_isim[multi], sizes[multi]), library_isim=lib_isim,
               scaffold_purity=purity(np.ones(C, bool)),
               medoid_isim_all=med_all, medoid_isim_nonsingleton=med_multi,
               tgt_molecules=int((tgt >= 0).sum()) if tgt is not None else np.nan)
    for r in REFS:
        row.update(ext_cols(r, ext.get(r)))

    top_rows = []
    for kt in KTOPS:
        sel, mem = top_members[kt]
        idx = rank[:kt]
        mi = idx[multi[idx]]
        tr = dict(k_top=kt, top_clusters=len(idx), top_molecules=int(mem.sum()), top_fraction=float(mem.sum() / n),
                  smallest_top_size=int(sizes[idx].min()),
                  isim=wmean(cl_isim[mi], sizes[mi]), scaffold_purity=purity(sel))
        ok = s_mode[idx] >= 0
        rec = s_cnt[idx[ok]] / ctx["scaf_count"][s_mode[idx[ok]]]
        tr["scaffold_recall"] = wmean(rec, sizes[idx[ok]])
        if tgt is not None:
            ok = t_mode[idx] >= 0
            rec = t_cnt[idx[ok]] / ctx["tgt_count"][t_mode[idx[ok]]]
            tr["target_recall"] = wmean(rec, sizes[idx[ok]])
            tr["tgt_molecules"] = int((mem & (tgt >= 0)).sum())
        else:
            tr["target_recall"] = np.nan; tr["tgt_molecules"] = np.nan
        for r in REFS:
            tr.update(ext_cols(r, ext_top[kt].get(r)))
        top_rows.append(tr)
    row["metrics_seconds"] = time.perf_counter() - t0
    row["gpu_seconds"] = t_gpu
    row["external_seconds"] = t_ext
    return row, top_rows


# ---------------------------------------------------------------- driver

def find_labels(ds):
    out = []
    for p in sorted((R.BP / "labels" / ds).glob("*/k*/s*.npy")):
        if p.name.endswith(".partial.npy"):
            continue
        mdir = p.parent.parent.name
        method, bf = (mdir.rsplit("_bf", 1)[0], int(mdir.rsplit("_bf", 1)[1])) if "_bf" in mdir else (mdir, R.BF)
        out.append(dict(path=p, method=mdir, base_method=method, bf=bf, k=float(p.parent.name[1:]),
                        seed=int(p.stem[1:])))
    return out


def write_csv(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


KEY = ["dataset", "method", "k", "seed"]


def run_dataset(ds, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    main_p, top_p = OUT / f"{ds}.csv", OUT / f"topk_{ds}.csv"
    main = pd.read_csv(main_p) if main_p.exists() and not force else pd.DataFrame()
    top = pd.read_csv(top_p) if top_p.exists() and not force else pd.DataFrame()
    done = {}
    for _, r in main.iterrows():
        done[(r.method, float(r.k), int(r.seed))] = (r.label_mtime, bool(r.butina_used))

    ctx = None
    for item in find_labels(ds):
        bpath = R.label_path(ds, "butina_matched", item["k"], 0)
        has_b = bpath.exists() and bpath.with_suffix(".json").exists()
        mtime = item["path"].stat().st_mtime
        key = (item["method"], item["k"], item["seed"])
        if key in done and abs(done[key][0] - mtime) < 1e-6 and done[key][1] == has_b:
            continue
        lab, info = load_labels(item["path"])
        if lab is None:
            continue
        if ctx is None:
            ctx = load_context(ds)
        if len(lab) != ctx["n"]:
            print(f"{ds} {key}: {len(lab)} labels != {ctx['n']} molecules, skipped", flush=True)
            continue
        butina = load_labels(bpath)[0] if has_b else None
        row, trows = evaluate(ctx, lab, info, butina)
        ident = dict(dataset=ds, method=item["method"], base_method=item["base_method"], bf=item["bf"], k=item["k"],
                     seed=item["seed"])
        row = {**ident, **row, "butina_used": butina is not None, "label_mtime": mtime}
        trows = [{**ident, **t} for t in trows]
        sel = lambda df: (df["method"] == item["method"]) & (df["k"] == item["k"]) & (df["seed"] == item["seed"])  # noqa
        if len(main):
            main = main[~sel(main)]
        if len(top):
            top = top[~sel(top)]
        main = pd.concat([main, pd.DataFrame([row])], ignore_index=True).sort_values(KEY, ignore_index=True)
        top = pd.concat([top, pd.DataFrame(trows)], ignore_index=True).sort_values(KEY + ["k_top"], ignore_index=True)
        write_csv(main, main_p); write_csv(top, top_p)
        print(f"{ds} {item['method']} k{item['k']:g} s{item['seed']}: {row['metrics_seconds']:.1f}s "
              f"(gpu {row['gpu_seconds']:.1f}, ext {row['external_seconds']:.1f}) clusters {row['clusters']} "
              f"isim {row['isim']:.4f} purity {row['scaffold_purity']:.4f} medoid {row['medoid_isim_nonsingleton']:.4f} "
              f"ami_scaf {row['scaf_ami']:.4f}", flush=True)


def load_context(ds):
    t0 = time.perf_counter()
    d = R.DATASETS[ds]
    fps = np.load(d["fps"])
    scaf = np.load(d["scaf"]).astype(np.int64)
    tgt = load_targets(ds, len(fps)) if ds in TARGET_SETS else None
    fps_gpu = torch.from_numpy(fps).to(DEV)
    ctx = dict(n=len(fps), fps_gpu=fps_gpu, scaf=scaf, scaf_count=np.bincount(scaf[scaf >= 0]), tgt=tgt,
               tgt_count=np.bincount(tgt[tgt >= 0]) if tgt is not None else None)
    print(f"{ds}: loaded {len(fps)} molecules in {time.perf_counter() - t0:.1f}s", flush=True)
    return ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(R.DATASETS))
    ap.add_argument("--force", action="store_true", help="recompute every row")
    a = ap.parse_args()
    torch.set_num_threads(min(32, os.cpu_count()))
    for ds in a.datasets:
        run_dataset(ds, a.force)
        torch.cuda.empty_cache() if DEV.type == "cuda" else None
