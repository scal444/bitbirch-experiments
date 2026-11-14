"""The 12.16B ZINC22 clustering next to three of the quality libraries, nvMolKit batched, k = 3.5, bf 1024.

(a) whole clustering: within-cluster iSIM, medoid iSIM (clusters of 2+), scaffold purity, library iSIM;
(b) within-cluster iSIM of the k_top largest clusters.
ZINC22: nvMolKit development branch (`bitbirch-gpu-perf`), one run; scaffold purity from the 500M-row scaffold
sample (`scaffold_purity_pooled_plugin`). Other libraries: nvMolKit batched labels, mean over input orders.
Data: data/zinc22/zinc22_overall.csv and zinc22_topk.csv (from the ZINC22 run) and the metrics CSVs.
Output: zinc22_vs_libraries.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import common as C  # noqa: E402

SETS = ["chembl34", "scu", "ena1m"]
# datasets (not methods; every bar is nvMolKit): ZINC22 in nvMolKit green, the smaller sets in a grey ramp, so none
# reads as another implementation's colour
COLORS = {"zinc22": C.NV, "chembl34": "#c9c7c0", "scu": C.NEUTRAL2, "ena1m": C.NEUTRAL3}
NAMES = {"zinc22": "ZINC22 (12.16 billion)", **C.DS_NAME}
METRICS = [("isim", "within-\ncluster iSIM"), ("medoid_isim_nonsingleton", "medoid\niSIM"),
           ("scaffold_purity", "scaffold\npurity"), ("library_isim", "library\niSIM")]


def data(k=3.5):
    g = C.P.ZINC22
    if not (g / "zinc22_overall.csv").exists():
        return None, None
    zo = pd.read_csv(g / "zinc22_overall.csv").iloc[0]
    zt = pd.read_csv(g / "zinc22_topk.csv")
    over = {"zinc22": {m: zo[m] for m, _ in METRICS if m in zo} | {"scaffold_purity": zo.scaffold_purity_pooled_plugin}}
    top = {"zinc22": zt.set_index("k_top")["isim"]}
    for ds in SETS:
        d = C.load_metrics(ds)
        d = d[(d.implementation == "nvmolkit_batched") & (d.k == k)]
        t = C.load_metrics(ds, topk=True)
        t = t[(t.implementation == "nvmolkit_batched") & (t.k == k)]
        if len(d):
            over[ds] = {m: d[m].mean() for m, _ in METRICS}
            top[ds] = t.groupby("k_top")["isim"].mean()
    return over, top


def main():
    C.style()
    over, top = data()
    if over is None:
        C.waiting("zinc22_vs_libraries", "ZINC22 metrics missing"); return
    keys = ["zinc22"] + [s for s in SETS if s in over]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(C.FULLW * 0.75, 2.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    w = 0.8 / len(keys)
    for i, key in enumerate(keys):
        x = np.arange(len(METRICS)) + (i - (len(keys) - 1) / 2) * w
        a1.bar(x, [over[key][m] for m, _ in METRICS], width=w * 0.9, color=COLORS[key], label=NAMES[key])
        s = top[key]
        a2.plot(s.index, s.values, marker="o", color=COLORS[key], label=NAMES[key])
    a1.set_xticks(range(len(METRICS)), [lab for _, lab in METRICS])
    a1.set_ylabel("value")
    a1.set_title("Whole clustering")
    a2.set_xscale("log")
    a2.set_xticks([5, 20, 100, 1000], ["5", "20", "100", "1000"])
    a2.minorticks_off()
    a2.set_xlabel("largest clusters taken")
    a2.set_ylabel("within-cluster iSIM")
    a2.set_title("Largest clusters")
    fig.suptitle("nvMolKit batched (GPU): ZINC22 12.16B against three smaller libraries, k = 3.5", y=1.03)
    fig.tight_layout()
    C.legend_below(fig, [a1], ncol=4, y=0.02)
    C.save(fig, "zinc22_vs_libraries")


if __name__ == "__main__":
    main()
