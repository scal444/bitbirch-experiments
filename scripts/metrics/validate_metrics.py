"""Check metrics.py against plain CPU / bblean / sklearn references on one labelling of a small set.

   python scripts/metrics/validate_metrics.py [DS] [IMPLEMENTATION] [K] [SEED]   (default: np bblean_serial 3.5 0)

IMPLEMENTATION: bblean_serial, bblean_split_merge (36 processes), bblean_batched or nvmolkit_batched; its labels
under BB_OUT/labels are used.
"""
import sys
import warnings
from collections import Counter

import numpy as np
from bblean.similarity import jt_isim_medoid, jt_isim_packed
from sklearn import metrics as skm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from metrics import P  # noqa: E402
from scripts.common.implementations import label_folder  # noqa: E402

warnings.filterwarnings("ignore")
ds, impl = (sys.argv[1:3] + ["np", "bblean_serial"][len(sys.argv[1:3]):])[:2]
k = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ctx = M.load_context(ds)
fps = np.load(P.DATASETS[ds]["fps"])
lab, info = M.load_labels(P.label_path(ds, label_folder(impl), k, seed))
row, trows = M.evaluate(ctx, lab, info)
codes, sizes = M.codes_of(lab)
C = len(sizes)
members = [[] for _ in range(C)]
for i, c in enumerate(codes):
    members[c].append(i)
scaf, tgt = ctx["scaf"], ctx["tgt"]
err = {}


def upd(name, a, b):
    e = abs(float(a) - float(b)) if not (np.isnan(a) and np.isnan(b)) else 0.0
    err[name] = max(err.get(name, 0.0), e)


# per-cluster iSIM and medoids straight from bblean, on a random sample of clusters + all small edge sizes
import torch  # noqa: E402
isim_t, med_t, _ = M.cluster_pass(ctx["fps_gpu"], torch.from_numpy(codes).to(M.DEV), sizes)
isim_g, med_g = isim_t.cpu().numpy(), med_t.cpu().numpy()
rng = np.random.default_rng(0)
multi = np.flatnonzero(sizes >= 2)
sample = np.unique(np.r_[rng.choice(multi, min(400, len(multi)), replace=False),
                         [np.flatnonzero(sizes == s)[0] for s in (1, 2, 3, 4) if (sizes == s).any()],
                         np.argsort(-sizes)[:20]])
med_bad = 0
for c in sample:
    rows = np.array(members[c])
    if len(rows) >= 2:
        upd("cluster iSIM vs bblean jt_isim_packed", isim_g[c], jt_isim_packed(fps[rows]))
med_bb = np.empty(C, np.int64)
for c in range(C):  # every cluster: bblean medoid on members in ascending row order
    rows = np.array(members[c])
    med_bb[c] = rows[jt_isim_medoid(fps[rows])[0]]
med_bad = int((med_bb != med_g).sum())
print(f"medoid mismatches vs bblean jt_isim_medoid: {med_bad} / {C} clusters")
upd("medoid iSIM non-singleton vs bblean medoids", row["medoid_isim_nonsingleton"], jt_isim_packed(fps[med_bb[sizes >= 2]]))

# whole-labelling numbers, plain references
ms = sizes >= 2
upd("isim (weighted)", row["isim"], sum(jt_isim_packed(fps[members[c]]) * sizes[c] for c in multi) / sizes[ms].sum())
upd("library iSIM vs bblean", row["library_isim"], jt_isim_packed(fps))
upd("library iSIM vs isim_sigma.json", row["library_isim"], P.isim(ds))
upd("medoid iSIM all", row["medoid_isim_all"], jt_isim_packed(fps[med_g]))
upd("medoid iSIM non-singleton", row["medoid_isim_nonsingleton"], jt_isim_packed(fps[med_g[ms]]))


def purity_ref(cl):
    top = tot = 0
    for c in cl:
        if sizes[c] < 2:
            continue
        s = [scaf[i] for i in members[c] if scaf[i] >= 0]
        if s:
            top += Counter(s).most_common(1)[0][1]; tot += len(s)
    return top / tot if tot else np.nan


def ext_ref(prefix, out, labs, ref):
    if len(labs) == 0:
        return
    upd(f"{prefix} ARI", out[f"{prefix}_ari"], skm.adjusted_rand_score(ref, labs))
    upd(f"{prefix} AMI", out[f"{prefix}_ami"], skm.adjusted_mutual_info_score(ref, labs))
    h, c, v = skm.homogeneity_completeness_v_measure(ref, labs)
    upd(f"{prefix} homogeneity", out[f"{prefix}_homogeneity"], h)
    upd(f"{prefix} completeness", out[f"{prefix}_completeness"], c)
    upd(f"{prefix} V-measure", out[f"{prefix}_v_measure"], v)


upd("scaffold purity", row["scaffold_purity"], purity_ref(range(C)))
ext_ref("scaf", row, codes, scaf)
if tgt is not None:
    m = tgt >= 0
    ext_ref("tgt", row, codes[m], tgt[m])
upd("counts: singletons", row["singletons"], sum(1 for c in range(C) if sizes[c] == 1))
upd("counts: largest", row["largest_cluster"], max(len(x) for x in members))

order = sorted(range(C), key=lambda c: (-sizes[c], c))
scount = Counter(scaf.tolist())
tcount = Counter(tgt[tgt >= 0].tolist()) if tgt is not None else None
for tr in trows:
    top = order[:tr["k_top"]]
    mem = np.zeros(len(codes), bool)
    for c in top:
        mem[members[c]] = True
    tm = [c for c in top if sizes[c] >= 2]
    upd("top-k iSIM", tr["isim"], sum(jt_isim_packed(fps[members[c]]) * sizes[c] for c in tm) / sum(sizes[c] for c in tm)
        if tm else np.nan)
    upd("top-k scaffold purity", tr["scaffold_purity"], purity_ref(top))
    num = den = 0.0
    for c in top:
        cnt = Counter(scaf[members[c]].tolist())
        best = min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        num += sizes[c] * best[1] / scount[best[0]]; den += sizes[c]
    upd("top-k scaffold recall", tr["scaffold_recall"], num / den)
    ext_ref("top-k scaf", {f"top-k scaf_{x}": tr[f"scaf_{x}"] for x in M.EXT}, codes[mem], scaf[mem])
    if tgt is not None:
        num = den = 0.0
        for c in top:
            t = [tgt[i] for i in members[c] if tgt[i] >= 0]
            if t:
                cnt = Counter(t)
                best = min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
                num += sizes[c] * best[1] / tcount[best[0]]; den += sizes[c]
        upd("top-k target recall", tr["target_recall"], num / den if den else np.nan)
        m2 = mem & (tgt >= 0)
        ext_ref("top-k tgt", {f"top-k tgt_{x}": tr[f"tgt_{x}"] for x in M.EXT}, codes[m2], tgt[m2])

print(f"{ds} {impl} k{k:g} s{seed}: {C} clusters; max abs error per metric:")
for k_, v in err.items():
    print(f"  {k_:45s} {v:.2e}")
