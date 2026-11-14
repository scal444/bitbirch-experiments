"""In the form of the BitBIRCH paper (2025) Fig 4: Calinski-Harabasz, Davies-Bouldin and Dunn index against threshold,
CHEMBL233_Ki, per implementation (mean +- sd over min_size when the data has a min_size column, as in the paper).
Data: moleculeace.load(). Output: moleculeace_indices_vs_threshold_<set>.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import moleculeace as M  # noqa: E402

C = M.C
IDX = [("ch", "Calinski-Harabasz (higher better)"), ("db", "Davies-Bouldin (lower better)"),
       ("dunn", "Dunn (higher better)")]


def main(name="CHEMBL233_Ki"):
    C.style()
    d = M.load()
    d = d[d["set"].astype(str).str.contains(name, case=False)] if len(d) else d
    idx = [i for i in IDX if len(d) and d[i[0]].notna().any()]
    if not idx:
        C.waiting("moleculeace_indices_vs_threshold", f"MoleculeACE CH/DB/Dunn for {name}"); return
    fig, axs = plt.subplots(1, len(idx), figsize=(C.FULLW, 1.9), squeeze=False)
    for ax, (col, lab) in zip(axs[0], idx):
        for p, st in M.QSTYLE.items():
            g = d[d.player == p].groupby("threshold")[col]
            mean, sd = g.mean(), g.std().fillna(0)
            if len(mean):
                ax.errorbar(mean.index, mean.values, yerr=sd.values, marker=st["marker"], color=st["color"],
                            label=st["label"], **M.mkw(p), capsize=1.2, elinewidth=0.7)
        ax.set_xlabel("threshold"); ax.set_ylabel(lab)
    fig.suptitle(f"Cluster validity indices vs threshold, {name}, four BitBIRCH implementations", y=1.02)
    fig.tight_layout()
    C.legend_below(fig, axs, ncol=4, y=0.02)
    C.save(fig, f"moleculeace_indices_vs_threshold_{name}")


if __name__ == "__main__":
    main()
