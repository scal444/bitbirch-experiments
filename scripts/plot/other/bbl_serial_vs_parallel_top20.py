"""In the form of BitBIRCH-Lean Fig 6 (a, b) and Table S3: sizes and within-cluster iSIM of the 20 largest clusters,
parallel vs serial (their 10M ZINC22 set, t 0.5). Here: the implementations in the quality comparison at k = 3.5,
bf 1024, one input order, on Enamine 1M (closest to their ZINC22 data), SCUBIDOO and ChEMBL34. The paper puts iSIM
on a second y axis; here it is a second row with the same x. Data: data/quality/top_clusters.csv
(scripts/metrics/top_clusters.py). Output: bbl_serial_vs_parallel_top20_<ds>.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import common as C  # noqa: E402

DATA = C.P.QUALITY / "top_clusters.csv"


def main(k=3.5):
    C.style()
    if not DATA.exists():
        C.waiting("bbl_serial_vs_parallel_top20", "run scripts/metrics/top_clusters.py"); return
    t = pd.read_csv(DATA)
    t = t[t.k == k]
    for ds in [d for d in ("ena1m", "scu", "chembl34") if d in set(t.dataset)]:
        d = t[t.dataset == ds]
        impls = C.implementations_present(d)
        fig, axs = plt.subplots(2, len(impls), figsize=(C.FULLW, 2.6), sharex=True, sharey="row", squeeze=False,
                                gridspec_kw=dict(height_ratios=[1.6, 1]))
        for j, p in enumerate(impls):
            s = d[d.implementation == p].sort_values("rank")
            st = C.QP[p]
            axs[0, j].bar(s["rank"], s["size"], width=0.75, color=st["color"])
            axs[1, j].plot(s["rank"], s["isim"], ls="none", marker=st["marker"], color=st["color"], ms=3.5)
            axs[0, j].set_title(C.panel_label(p), fontsize=6.5)
            axs[1, j].set_xlabel("cluster rank")
            axs[1, j].set_xticks([1, 5, 10, 15, 20])
            axs[1, j].set_ylim(0, 1)
        axs[0, 0].set_ylabel("molecules")
        axs[1, 0].set_ylabel("within-cluster\niSIM")
        fig.align_ylabels(axs[:, 0])
        fig.suptitle(f"20 largest clusters, {C.DS_NAME[ds]}, k = {k:g} (bf 1024, one input order)", y=0.99)
        fig.tight_layout(h_pad=0.4)
        C.save(fig, f"bbl_serial_vs_parallel_top20_{ds}")


if __name__ == "__main__":
    main()
