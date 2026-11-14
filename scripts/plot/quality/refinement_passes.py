"""Refinement experiment: after bblean's refinement, do the implementations reach the same quality?

Per dataset: within-cluster iSIM, scaffold purity, scaffold AMI and target AMI (ChEMBL34, ChEMBL35 additions) against the
number of clusters, one line per implementation along the refinement passes (pass 0 = unrefined, open marker; then
recluster passes). Extended figures: input order 0; k = 3.5, bf 1024. Procedures: rc0 = recluster_inplace(extra_threshold=0) x10,
rc1 = extra_threshold sigma x5, ref = refine_inplace (largest cluster) x5; one figure each.
Data: data/refinement/quality_<ds>.csv (else BB_REFINE_CSV). The split-merge runs
there use 36 parts / 36 processes (BB_REFINE_MR_P overrides the label if a different P was run).
Output: refinement_passes_<proc>_<ds|all>.* (input order 0, against the number of clusters).
refinement_passes_all.* -- every library (rows) and the seven measures of quality_chembl34 (columns) against the recluster pass, mean and 95 % interval over all input orders, for bblean serial,
split-merge and bblean batched. Refinement runs in bblean on the clustering's tree, so nvMolKit batched (identical to
bblean batched before refinement) has no separate refined series.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import common as C  # noqa: E402
from quality import PANELS  # noqa: E402

PLAYER = {"serial": "serial", "batched": "bblean_batched",  # refinement batched runs used bblean batched (multithreaded CPU)
          "parallel": "multiround", "multiround": "multiround"}
COLS = [("isim", "within-cluster iSIM"), ("scaffold_purity", "scaffold purity"), ("scaf_ami", "AMI vs scaffolds"),
        ("tgt_ami", "AMI vs targets")]
PROC = {"rc0": "recluster x10", "rc1": r"recluster (+$\sigma$) x5", "ref": "refine largest x5"}


def load():
    src = C.P.refine_csv()
    frames = [pd.read_csv(f) for f in sorted(src.glob("quality_*.csv"))]
    if not frames:
        return pd.DataFrame(), src
    d = pd.concat(frames, ignore_index=True)
    d["player"] = d.player.map(PLAYER)
    return d[d.k == 3.5], src


def labels(src):
    lab = {p: dict(C.QP[p]) for p in C.QP}
    if C.refine_mr_p() != C.MAIN_P:  # the provisional 10 parts / 10 processes command
        lab["multiround"]["label"] = f"{C.PAR_NAME} (provisional, P={C.refine_mr_p()})"
    return lab


def draw(axs_row, d, ds, cols, lab):
    for ax, (col, ylab) in zip(axs_row, cols):
        for p in ("serial", "multiround", "bblean_batched"):
            s = d[(d.dataset == ds) & (d.player == p)].sort_values("it")
            if s.empty or s[col].isna().all():
                continue
            st = lab[p]
            ax.plot(s.clusters / 1e3, s[col], color=st["color"], marker=st["marker"], ms=3, label=st["label"])
            ax.plot(s.clusters.iloc[:1] / 1e3, s[col].iloc[:1], ls="none", marker=st["marker"], ms=5,
                    markerfacecolor="white", color=st["color"])
        C.logx_plain(ax)
        ax.set_ylabel(ylab)


def si_figure(d):
    """Supplement: rc0 passes, every library and measure, mean and 95 % interval over input orders."""
    q = d[d.proc == "rc0"]
    dss = [ds for ds in C.DS_ORDER if ds in set(q.dataset)]
    players = ["serial", "bblean_batched", "multiround"]
    fig, axs = plt.subplots(len(dss), len(PANELS), figsize=(C.FULLW, 1.2 * len(dss) + 0.6), squeeze=False)
    its = sorted(q.it.unique())
    for r, ds in enumerate(dss):
        g = q[q.dataset == ds]
        for c, (col, title, sc) in enumerate(PANELS):
            ax = axs[r, c]
            if not g[col].notna().any():
                ax.axis("off"); continue
            for p in players:
                st = [C.mean_ci(np.asarray(g[(g.player == p) & (g.it == it)][col] * sc, float)) for it in its]
                m = np.array([t[0] for t in st]); lo = np.array([t[1] for t in st]); hi = np.array([t[2] for t in st])
                if (g[col].dropna() >= 0).all():
                    lo = np.maximum(lo, 0)
                col_ = C.QP[p]["color"]
                ax.fill_between(its, lo, hi, color=col_, alpha=0.18, lw=0, zorder=1)
                ax.plot(its, m, color=col_, lw=0.9, marker="o", ms=2, zorder=2)
                ax.plot(its[:1], m[:1], ls="none", marker="o", ms=3.5, mfc="white", color=col_, zorder=3)
            ax.set_xticks([0, 5, 10], ["0", "5", "10"] if r == len(dss) - 1 else [])
            ax.grid(axis="x", visible=False)
            ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(title.replace("Within-cluster ", "Within-cluster\n").replace(", ", ",\n"), fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{C.DS_NAME[ds]}\n(n = {g.seed.nunique()})", fontsize=6.5, fontweight="bold")
    for c in range(len(PANELS)):  # pass labels on the lowest visible panel of each column
        vis = [r for r in range(len(dss)) if axs[r, c].axison]
        if vis:
            axs[vis[-1], c].set_xticks([0, 5, 10], ["0", "5", "10"]); axs[vis[-1], c].set_xlabel("pass", fontsize=6)
    hs = [Line2D([], [], marker="o", lw=0.8, ms=4, color=C.QP[p]["color"], label=C.QP[p]["label"]) for p in players]
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.5, w_pad=0.3)
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    C.save(fig, "refinement_passes_all")


def main():
    C.style()
    d, src = load()
    if d.empty:
        C.waiting("refinement_passes", f"no quality_*.csv in {src}"); return
    lab = labels(src)
    si_figure(d)
    d = d[d.seed == 0]
    for proc in ("rc0", "rc1", "ref"):
        q = d[(d.proc == proc) | (d.it == 0)]
        q = q[q.proc == proc]
        # one figure per dataset and one with every dataset
        dss = [ds for ds in C.DS_ORDER if ds in set(q.dataset)]
        for ds in dss:
            cols = [c for c in COLS if q[q.dataset == ds][c[0]].notna().any()]
            fig, axs = plt.subplots(1, len(cols), figsize=(C.FULLW, 1.85), squeeze=False)
            draw(axs[0], q, ds, cols, lab)
            for ax in axs[0]:
                ax.set_xlabel("clusters (thousands)")
            fig.suptitle(f"Quality along bblean refinement ({PROC[proc]}), {C.DS_NAME[ds]}, k = 3.5", y=1.04)
            fig.tight_layout()
            C.legend_below(fig, axs, ncol=3, y=0.02)
            C.save(fig, f"refinement_passes_{proc}_{ds}")
        if proc == "rc0":
            continue  # every library for rc0: si_figure() (refinement_passes_all)
        fig, axs = plt.subplots(len(dss), len(COLS), figsize=(C.FULLW, 1.45 * len(dss) + 0.4), squeeze=False)
        for i, ds in enumerate(dss):
            draw(axs[i], q, ds, COLS, lab)
            for ax in axs[i]:
                ax.set_title(C.DS_NAME[ds], fontsize=7)
            if q[q.dataset == ds].tgt_ami.isna().all():
                axs[i, 3].set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel("clusters (thousands)")
        fig.suptitle(f"Quality along bblean refinement ({PROC[proc]}), k = 3.5, input order 0", y=1.0)
        fig.tight_layout(h_pad=0.8)
        fig.align_ylabels(axs[:, 0])
        C.legend_below(fig, axs, ncol=3, y=0.005)
        C.save(fig, f"refinement_passes_{proc}_all")


if __name__ == "__main__":
    main()
