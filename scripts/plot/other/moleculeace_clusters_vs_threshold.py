"""In the form of the BitBIRCH paper (2025) Fig 2: number of clusters with at least min_size members (1, 2, 3, 5, 10)
against threshold, CHEMBL262_Ki, for each implementation. One panel per min_size, one line per
implementation along the threshold.
Data: moleculeace.load(). Output: moleculeace_clusters_vs_threshold_<set>.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import moleculeace as M  # noqa: E402

C = M.C
MINS = (1, 2, 3, 5, 10)


def main(name="CHEMBL262_Ki"):
    C.style()
    d = M.load()
    d = d[d["set"].astype(str).str.contains(name, case=False)] if len(d) else d
    mins = [m for m in MINS if (d.min_size == m).any()] if len(d) else []
    if not mins:
        C.waiting("moleculeace_clusters_vs_threshold", f"MoleculeACE cluster counts by min_size for {name}"); return
    fig, axs = plt.subplots(1, len(mins), figsize=(C.FULLW, 1.8), squeeze=False)
    for ax, m in zip(axs[0], mins):
        for p, st in M.QSTYLE.items():
            s = d[(d.player == p) & (d.min_size == m)].groupby("threshold").clusters_kept.mean()
            if len(s):
                ax.plot(s.index, s.values, marker=st["marker"], color=st["color"], label=st["label"], **M.mkw(p))
        ax.set_title(f"{m}+ members", fontsize=7)
        ax.set_xlabel("threshold")
    axs[0, 0].set_ylabel("clusters with at least\nthat many members")
    fig.suptitle(f"Cluster counts vs threshold, {name}, four BitBIRCH implementations", y=1.02)
    fig.tight_layout()
    C.legend_below(fig, axs, ncol=4, y=0.02)
    C.save(fig, f"moleculeace_clusters_vs_threshold_{name}")


if __name__ == "__main__":
    main()
