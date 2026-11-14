"""In the form of the BitBIRCH best-practices paper's Fig 5 (and S17 for SCUBIDOO): sizes and unique Murcko scaffolds of the
20 largest clusters, plus the number of clusters above each size, per implementation, k in {3, 3.5, 4}, input order 0,
bf 1024.

Data: data/quality/top_clusters.csv, size_bins.csv (scripts/metrics/top_clusters.py).
Output: top20_clusters_<ds>_k<k>.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import common as C  # noqa: E402

DATA = C.P.QUALITY
BINS = (0, 1, 10, 100, 1000)


def main():
    C.style()
    if not (DATA / "top_clusters.csv").exists():
        C.waiting("top20_clusters", "run scripts/metrics/top_clusters.py"); return
    top, bins = pd.read_csv(DATA / "top_clusters.csv"), pd.read_csv(DATA / "size_bins.csv")
    from matplotlib.patches import Patch
    for (ds, k), t in top.groupby(["dataset", "k"]):
        impls = C.implementations_present(t)
        n = len(impls)
        fig = plt.figure(figsize=(C.FULLW, 3.9))
        gs = fig.add_gridspec(2, n, height_ratios=[1, 0.8], hspace=0.8, wspace=0.3)
        ymax = t["size"].max() * 1.05
        axs = []
        for i, p in enumerate(impls):
            ax = fig.add_subplot(gs[0, i], sharey=axs[0] if axs else None)
            axs.append(ax)
            s = t[t.implementation == p].sort_values("rank")
            ax.bar(s["rank"], s["size"], width=0.8, color=C.QP[p]["color"], alpha=0.35)
            ax.bar(s["rank"], s["unique_scaffolds"], width=0.5, color=C.QP[p]["color"])
            ax.set_title(C.panel_label(p), fontsize=6.5,
                         color=C.QP[p]["color"] if p != "bblean_split_merge" else C.BB_SER_REF)
            ax.set_xlabel("cluster rank")
            ax.set_ylim(0, ymax)
            ax.set_xticks([1, 5, 10, 15, 20])
            if i:
                ax.tick_params(labelleft=False)
        axs[0].set_ylabel("molecules")
        fig.legend([Patch(color="0.45", alpha=0.35), Patch(color="0.45")], ["members", "unique scaffolds"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.49), ncol=2)
        ax = fig.add_subplot(gs[1, :])
        b = bins[(bins.dataset == ds) & (bins.k == k)]
        w = 0.8 / max(n, 1)
        for i, p in enumerate(impls):
            r = b[b.implementation == p]
            if len(r):
                vals = [r[f"gt{x}"].iloc[0] for x in BINS]
                ax.bar(np.arange(len(BINS)) + (i - (n - 1) / 2) * w, vals, width=w * 0.9,
                       color=C.QP[p]["color"], label=C.QP[p]["label"])
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
        C.plain_log(ax, ticks=(1, 10, 100, 1000, 10000, 100000, 1000000))
        ax.set_xticks(range(len(BINS)), [f"> {x:,}" for x in BINS])
        ax.set_xlabel("cluster size (members)")
        ax.set_ylabel("clusters")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"20 largest clusters and clusters above each size, {C.DS_NAME[ds]}, k = {k:g}", y=1.03)
        C.save(fig, f"top20_clusters_{ds}_k{k:g}")


if __name__ == "__main__":
    main()
