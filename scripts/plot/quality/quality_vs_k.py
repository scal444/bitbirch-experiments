"""Quality against the threshold multiplier k (t = iSIM + k sigma), per implementation (form of the BitBIRCH best-practices
paper's Fig 2). One dot per input order (bblean serial hollow), black mean line, 95 % CI whisker; the means of one
implementation are joined along k. One figure per library: quality_vs_k_<ds> (no target
panel outside ChEMBL). quality_vs_k: every library in one figure (rows =
libraries, columns = measures), mean and 95 % interval over input orders, means joined along k.
bf 1024. Data: metrics CSVs (scripts/metrics/metrics.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import common as C  # noqa: E402

PANELS = [("clusters", "Clusters (thousands)", 1e-3), ("isim", "Within-cluster iSIM ↑", 1),
          ("scaffold_purity", "Scaffold purity ↑", 1), ("scaf_ami", "AMI vs scaffolds ↑", 1),
          ("tgt_ami", "AMI vs targets ↑", 1)]
W = 0.12


def main():
    C.style()
    if not C.any_metrics(topk=False):
        C.waiting("quality_vs_k", "per-run metrics in data/quality/by_run"); return
    for ds in C.DS_ORDER:
        d = C.load_metrics(ds, bf=C.P.BF)
        d = d[d.player.isin(list(C.QP)) & d.k.isin(list(C.P.KS))]
        if d.empty:
            continue
        panels = [p for p in PANELS if d[p[0]].notna().any()]
        players = C.players_present(d)
        fig, axs = plt.subplots(1, len(panels), figsize=(C.FULLW, 1.9), squeeze=False)
        rng = np.random.default_rng(0)
        for ax, (col, title, sc) in zip(axs[0], panels):
            for j, p in enumerate(players):
                st = C.QP[p]
                xs, ms = [], []
                for k in C.P.KS:
                    x = k + (j - (len(players) - 1) / 2) * W
                    ms.append(C.strip(ax, x, d[(d.player == p) & (d.k == k)][col] * sc, st["color"],
                                      hollow=st["hollow"], width=W * 0.9, s=6, rng=rng))
                    xs.append(x)
                ax.plot(xs, ms, color=st["color"], lw=0.8, zorder=1)
            ax.set_xticks(list(C.P.KS), [f"{k:g}" for k in C.P.KS])
            ax.set_xlabel("k"); ax.set_title(title, fontsize=7)
        hs = [Line2D([], [], marker="o", ls="none", color=C.QP[p]["color"], ms=4,
                     mfc="none" if C.QP[p]["hollow"] else C.QP[p]["color"], label=C.QP[p]["label"]) for p in players]
        fig.suptitle(f"{C.DS_NAME[ds]}, threshold iSIM + kσ", x=0.01, y=0.99, ha="left", fontsize=8.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=0.6)
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, 0.0))
        C.save(fig, f"quality_vs_k_{ds}")


from quality import PANELS as SI_PANELS  # noqa: E402  (the seven measures of quality_chembl34)
SI_W = 0.1


def all_libraries():
    """Every library in one figure, mean and 95 % interval per implementation and k."""
    frames = {ds: C.load_metrics(ds, bf=C.P.BF) for ds in C.DS_ORDER}
    frames = {ds: d[d.player.isin(list(C.QP)) & d.k.isin(list(C.P.KS))] for ds, d in frames.items() if len(d)}
    dss = [ds for ds in C.DS_ORDER if ds in frames and len(frames[ds])]
    if not dss:
        return
    players = ["serial", "bblean_batched", "batched", "multiround"]
    fig, axs = plt.subplots(len(dss), len(SI_PANELS), figsize=(C.FULLW, 1.25 * len(dss) + 0.5), squeeze=False)
    for r, ds in enumerate(dss):
        d = frames[ds]
        for c, (col, title, sc) in enumerate(SI_PANELS):
            ax = axs[r, c]
            if not d[col].notna().any():
                ax.axis("off"); continue
            for j, p in enumerate(players):
                xs, ms = [], []
                for k in C.P.KS:
                    x = k + (j - (len(players) - 1) / 2) * SI_W
                    ms.append(C.mean_point(ax, x, d[(d.player == p) & (d.k == k)][col] * sc, C.QP[p]["color"], ms=3.5))
                    xs.append(x)
                ax.plot(xs, ms, color=C.QP[p]["color"], lw=0.8, zorder=1)
            ax.set_xticks(list(C.P.KS), [f"{k:g}" for k in C.P.KS] if r == len(dss) - 1 else [])
            ax.grid(axis="x", visible=False)
            ax.tick_params(labelsize=5.5)
            if r == 0:
                ax.set_title(title.replace("Within-cluster ", "Within-cluster\n").replace(", ", ",\n"), fontsize=6.5)
            if r == len(dss) - 1:
                ax.set_xlabel("k", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{C.DS_NAME[ds]}\n(n = {d[d.k == 3.5].seed.nunique()} at k = 3.5)", fontsize=6.5,
                              fontweight="bold")
    for c in range(len(SI_PANELS)):  # k labels on the lowest visible panel of each column
        vis = [r for r in range(len(dss)) if axs[r, c].axison]
        if vis:
            axs[vis[-1], c].set_xticks(list(C.P.KS), [f"{k:g}" for k in C.P.KS]); axs[vis[-1], c].set_xlabel("k", fontsize=6.5)
    hs = [Line2D([], [], marker="o", ls="-", lw=0.8, color=C.QP[p]["color"], ms=4, label=C.QP[p]["label"])
          for p in players]
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=0.5)
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    C.save(fig, "quality_vs_k")


if __name__ == "__main__":
    main()
    all_libraries()
