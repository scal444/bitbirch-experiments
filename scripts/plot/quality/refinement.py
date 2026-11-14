"""Quality on ChEMBL34 before and after bblean's refinement (10 recluster passes, recluster_inplace with
extra_threshold 0), per implementation, k = 3.5, over the input orders that were refined.

Linear axes, x = implementation: open point = mean before refinement, filled point = mean after, each with its 95 %
interval over input orders, and an arrow in the method colour from before to after. The batched series is bblean
batched (multithreaded CPU), which made and refined those clusterings. Every library and every pass:
refinement_passes_*.
Data: data/refinement/quality_chembl34.csv (scripts/refinement). Output: plots/quality/.../refinement_chembl34.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import common as C  # noqa: E402
from quality import MAIN_DS, PANELS  # noqa: E402

PLAYER = {"serial": "serial", "batched": "batched", "parallel": "multiround", "multiround": "multiround"}
PLAYERS = ["serial", "batched", "multiround"]
PROC, ITER = "rc0", 10
DX = 0.17  # before at j - DX, after at j + DX


def main():
    C.style()
    f = C.P.refine_csv() / f"quality_{MAIN_DS}.csv"
    if not f.exists():
        C.waiting("refinement_chembl34", f"no {f}"); return
    d = pd.read_csv(f)
    d["player"] = d.player.map(PLAYER)
    d = d[(d.k == 3.5) & (d.proc == PROC) & d.it.isin([0, ITER])]
    # the batched clusterings here were made and refined with bblean batched (multithreaded CPU)
    style = {"serial": (C.BB_SER, "bblean serial"), "batched": (C.OURS, C.OURS_NAME),
             "multiround": (C.BB_PAR, C.PAR_NAME)}
    fig, axs = plt.subplots(2, 4, figsize=(C.FULLW, 3.4))
    for ax, (col, title, sc) in zip(axs.flat, PANELS):
        for j, p in enumerate(PLAYERS):
            colr = style[p][0]
            ms = [C.mean_point(ax, j + dx, d[(d.player == p) & (d.it == it)][col] * sc, colr, hollow=hollow)
                  for it, dx, hollow in ((0, -DX, True), (ITER, DX, False))]
            if all(np.isfinite(ms)):
                ax.annotate("", (j + DX, ms[1]), (j - DX, ms[0]),
                            arrowprops=dict(arrowstyle="-|>", color=colr, lw=0.8, mutation_scale=7,
                                            shrinkA=4, shrinkB=4), zorder=2)
        ax.set_xticks([]); ax.set_xlim(-0.6, len(PLAYERS) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=7)
    hs = [Line2D([], [], marker="o", ls="none", color=style[p][0], ms=5, label=style[p][1]) for p in PLAYERS]
    hs += [Line2D([], [], marker="o", ls="none", color="0.35", mfc="white", mew=1.2, ms=5, label="before refinement"),
           Line2D([], [], marker="o", ls="none", color="0.35", ms=5, label=f"after {ITER} recluster passes")]
    axs.flat[-1].axis("off")
    axs.flat[-1].legend(handles=hs, loc="center", fontsize=6.5, frameon=False, title=f"{C.DS_NAME[MAIN_DS]}, k = 3.5",
                        title_fontsize=7)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    C.save(fig, "refinement_chembl34")


if __name__ == "__main__":
    main()
