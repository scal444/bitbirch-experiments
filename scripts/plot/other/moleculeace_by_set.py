"""MoleculeACE at threshold 0.65: number of clusters and the three clustering indices (Calinski-Harabasz,
Davies-Bouldin, Dunn; computed on clusters of at least 2 members) for each implementation, as the mean over the 30
sets with its 95 % interval (same form as quality_chembl34). No implementation is used as a reference.
Settings of the BitBIRCH paper: branching factor 50, input order 0, split-merge with 10 parts and 10 processes.
A measure is averaged only over the sets where every implementation has it: Davies-Bouldin is undefined for
some sets (two clusters with the same centre), so it uses 25 of the 30 sets.
Data: moleculeace.load() (data/moleculeace/moleculeace_indices.csv).
Output: moleculeace_by_set.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import moleculeace as M  # noqa: E402

C = M.C
PANELS = [("clusters", "Clusters"), ("ch", "Calinski–Harabasz ↑"), ("db", "Davies–Bouldin ↓"), ("dunn", "Dunn ↑")]
IMPLS = ["bblean_serial", "bblean_batched", "nvmolkit_batched", "bblean_split_merge"]


def label(p):
    return M.QSTYLE[p]["label"] if p in M.QSTYLE else C.QP[p]["label"]


def main(t=0.65):
    C.style()
    d = M.load()
    d = d[np.isclose(d.threshold.astype(float), t) & (d.min_size == 2)] if len(d) else d
    if d.empty:
        C.waiting("moleculeace_by_set", "MoleculeACE runs at threshold 0.65"); return
    d = d.groupby(["set", "implementation"])[[c for c, _ in PANELS]].mean().reset_index()
    fig, axs = plt.subplots(1, len(PANELS), figsize=(C.FULLW, 1.9))
    for ax, (col, title) in zip(axs, PANELS):
        w = d[d.implementation.isin(IMPLS)].pivot(index="set", columns="implementation", values=col)[IMPLS].dropna()
        for j, p in enumerate(IMPLS):  # only sets where every implementation has a value, so the means compare
            C.mean_point(ax, j, w[p], C.QP[p]["color"])
        ax.set_xticks([]); ax.set_xlim(-0.6, len(IMPLS) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(title if len(w) == d["set"].nunique() else f"{title} ({len(w)} sets)", fontsize=7)
    hs = [Line2D([], [], ls="none", marker="o", ms=5, color=C.QP[p]["color"], label=label(p)) for p in IMPLS]
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.8)
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    C.save(fig, "moleculeace_by_set")


if __name__ == "__main__":
    main()
