"""The k_top largest clusters (k_top = 5, 20, 100, 1000): within-cluster iSIM, scaffold purity, scaffold recall
(and target recall on ChEMBL34 / ChEMBL35 additions), per implementation, k = 3.5, mean over input orders with min-max bars.
Data: topk_<ds>.csv metrics. Output: topk_<ds>.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import common as C  # noqa: E402

COLS = [("isim", "within-cluster iSIM"), ("scaffold_purity", "scaffold purity"),
        ("scaffold_recall", "scaffold recall"), ("target_recall", "target recall")]


def main(k=3.5):
    C.style()
    if not C.any_metrics(topk=True):
        C.waiting("topk", "per-run metrics in data/quality/by_run"); return
    for ds in C.DS_ORDER:
        d = C.load_metrics(ds, topk=True)
        d = d[d.implementation.isin(list(C.QP)) & (d.k == k)]
        if d.empty:
            continue
        cols = [c for c in COLS if d[c[0]].notna().any()]
        a = C.agg_seeds(d, [c[0] for c in cols], by=("implementation", "k_top"))
        fig, axs = plt.subplots(1, len(cols), figsize=(C.FULLW * len(cols) / 4, 1.8), squeeze=False)
        for ax, (col, lab) in zip(axs[0], cols):
            for p in C.implementations_present(a):
                s = a[a.implementation == p].sort_values("k_top")
                C.errbar(ax, s.k_top, s[col + "_mean"], s[col + "_min"], s[col + "_max"], p)
            ax.set_xscale("log")
            ax.set_xticks([5, 20, 100, 1000], ["5", "20", "100", "1000"])
            ax.minorticks_off()
            ax.set_xlabel("largest clusters taken")
            ax.set_ylabel(lab)
        fig.suptitle(f"The largest clusters, {C.DS_NAME[ds]}, k = {k:g}", y=1.03)
        fig.tight_layout()
        C.legend_below(fig, axs, ncol=2, y=0.02)
        C.save(fig, f"topk_{ds}")


if __name__ == "__main__":
    main()
