"""Cluster quality of each implementation on ChEMBL34 (representative library), k = 3.5, no refinement.

Panels (linear axes, x = implementation, mean and 95 % interval over input orders): within-cluster iSIM, scaffold
purity, singletons; AMI and ARI with Murcko scaffolds and with ChEMBL single-target annotations.
Output (plots/quality/ of the results folder): quality_chembl34; quality_by_library (the same panels for every
library); clusters_by_library (number of clusters per library); quality_all_scores (AMI, ARI, homogeneity,
completeness and V-measure against scaffolds and targets).
Four series: bblean serial, bblean batched (multithreaded CPU), nvMolKit batched (GPU), bblean split-merge. The two batched
implementations give identical labels on every setting (scripts/run/check_batched_vs_nvmolkit.py).
Data: C.load_metrics (data/quality/by_run/ and by_run_earlier_labels/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import common as C  # noqa: E402

IMPLS = ["bblean_serial", "bblean_batched", "nvmolkit_batched", "bblean_split_merge"]
MAIN_DS = "chembl34"  # representative library: the only one with both scaffolds and targets at full size
PANELS = [("isim", "Within-cluster iSIM ↑", 1),
          ("scaffold_purity", "Scaffold purity ↑", 1), ("singleton_fraction", "Singletons (%)", 100),
          ("scaf_ami", "AMI, scaffolds ↑", 1), ("scaf_ari", "ARI, scaffolds ↑", 1),
          ("tgt_ami", "AMI, targets ↑", 1), ("tgt_ari", "ARI, targets ↑", 1)]
# Supplementary: every agreement score against each reference labelling
REFS = [("scaf", "Murcko scaffolds"), ("tgt", "ChEMBL targets")]
SCORES = [("ami", "AMI"), ("ari", "ARI"), ("homogeneity", "homogeneity"), ("completeness", "completeness"),
          ("v_measure", "V-measure")]
K = 3.5
W = 0.19


def strips(ax, d, dss, col, sc, rng):
    shown = [ds for ds in dss if d[(d.dataset == ds)][col].notna().any()]
    for i, ds in enumerate(shown):
        for j, p in enumerate(IMPLS):
            v = d[(d.dataset == ds) & (d.implementation == p)][col] * sc
            C.strip(ax, i + (j - 1.5) * W, v, C.QP[p]["color"], width=W * 0.8, s=6, rng=rng)
    ax.set_xticks(range(len(shown)), [C.DS_NAME[ds] for ds in shown], fontsize=5.5, rotation=40, ha="right",
                  rotation_mode="anchor")
    ax.set_xlim(-0.55, len(shown) - 0.45)
    ax.set_ylim(0, None)


def one_library(ax, d, ds, col, sc):
    """x = implementation, mean and 95 % interval over input orders, linear axis."""
    for j, p in enumerate(IMPLS):
        C.mean_point(ax, j, d[(d.dataset == ds) & (d.implementation == p)][col] * sc, C.QP[p]["color"])
    ax.set_xticks([]); ax.set_xlim(-0.6, len(IMPLS) - 0.4)
    ax.grid(axis="x", visible=False)


def handles():
    return [Line2D([], [], marker="o", ls="none", color=C.QP[p]["color"], ms=5, label=C.QP[p]["label"])
            for p in IMPLS]


def legend(fig):
    fig.legend(handles=handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.5,
               columnspacing=1.0)


def main():
    C.style()
    frames = [C.load_metrics(ds) for ds in C.DS_ORDER]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if d.empty:
        C.waiting("quality_chembl34", "per-run metrics in data/quality/by_run"); return
    d = d[d.implementation.isin(IMPLS) & (d.k == K)]
    dss = [ds for ds in C.DS_ORDER if ds in set(d.dataset)]
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 4, figsize=(C.FULLW, 3.4))  # ChEMBL34; legend in the eighth slot
    for ax, (col, title, sc) in zip(axs.flat, PANELS):
        one_library(ax, d, MAIN_DS, col, sc)
        ax.set_title(title, fontsize=7)
    axs.flat[-1].axis("off")
    axs.flat[-1].legend(handles=handles(), loc="center", fontsize=6.5, frameon=False, title="ChEMBL34, k = 3.5",
                        title_fontsize=7)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    C.save(fig, "quality_chembl34")

    # every library, as quality_chembl34 (rows = measures, columns = libraries, each panel its own y range)
    fig, axs = plt.subplots(len(PANELS), len(dss), figsize=(C.FULLW, 1.05 * len(PANELS) + 0.5))
    for r, (col, title, sc) in enumerate(PANELS):
        for c, ds in enumerate(dss):
            ax = axs[r, c]
            if not d[d.dataset == ds][col].notna().any():
                ax.axis("off"); continue
            one_library(ax, d, ds, col, sc)
            ax.tick_params(labelsize=5.5)
            if r == 0:
                ax.set_title(C.DS_NAME[ds], fontsize=7)
            if c == 0:
                ax.set_ylabel(title.replace(" ↑", "").replace(", ", "\n"), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=0.5)
    legend(fig)
    C.save(fig, "quality_by_library")

    fig, axs = plt.subplots(1, len(dss), figsize=(C.FULLW, 2.1))  # cluster counts
    for ax, ds in zip(axs, dss):
        one_library(ax, d, ds, "clusters", 1e-3)
        ax.set_ylim(0, None)
        ax.set_title(C.DS_NAME[ds], fontsize=7)
    axs[0].set_ylabel("clusters (thousands)")
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    legend(fig)
    C.save(fig, "clusters_by_library")

    fig, axs = plt.subplots(len(REFS), len(SCORES), figsize=(C.FULLW, 5.6 * 2 / 3))
    for r, (ref, rname) in enumerate(REFS):
        for c, (sc_, sname) in enumerate(SCORES):
            ax = axs[r, c]
            strips(ax, d, dss, f"{ref}_{sc_}", 1, rng)
            ax.set_title(f"{sname} ↑", fontsize=7)
            if c == 0:
                ax.set_ylabel(rname, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0, w_pad=0.4)
    legend(fig)
    C.save(fig, "quality_all_scores")


if __name__ == "__main__":
    main()
