"""Working figures from data/refinement/agreement_<ds>.csv and data/refinement/quality_<ds>.csv -> BB_REFINE_POST_LABELS/figures/*.png (working figures, outside git)

   python scripts/refinement/plot.py [--paths cfg.json] [--datasets c35,np,scu,ena1m,chembl34]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NAMES = {"c35": "ChEMBL35 additions", "np": "ChEMBL NP", "scu": "SCUBIDOO", "ena1m": "Enamine 1M", "chembl34": "ChEMBL34"}
PLAYER = {"serial": "bblean serial", "parallel": "bblean parallel", "batched": "nvMolKit batched"}
PCOL = {"serial": "#eb6834", "parallel": "#f4a582", "batched": "#76B900"}  # house colours (colour = method)
PAIRS = [("serial", "batched"), ("serial", "parallel"), ("parallel", "batched")]
PAIRCOL = {("serial", "batched"): "#76B900", ("serial", "parallel"): "#f4a582", ("parallel", "batched"): "#4b6a88"}
PROC = {"rc0": "recluster, +0", "rc1": "recluster, +1σ per pass", "ref": "split largest + refit"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load(cfg, dss):
    out = Path(cfg["csv_dir"])
    a = pd.concat([pd.read_csv(out / f"agreement_{d}.csv") for d in dss if (out / f"agreement_{d}.csv").exists()])
    q = pd.concat([pd.read_csv(out / f"quality_{d}.csv") for d in dss if (out / f"quality_{d}.csv").exists()])
    return a, q


def fig_players(a, dss, path):
    """Between-player AMI by iteration (seed-mean), with the same-player different-seed AMI as the reference."""
    procs = list(C.PROCS)
    fig, axs = plt.subplots(len(procs), len(dss), figsize=(2.6 * len(dss), 2.2 * len(procs)), sharey=True,
                            squeeze=False)
    for j, ds in enumerate(dss):
        d = a[a.dataset == ds]
        for i, proc in enumerate(procs):
            ax = axs[i, j]
            p = d[(d.kind == "players") & (d.proc == proc)]
            for pa, pb in PAIRS:
                s = p[(p.player_a == pa) & (p.player_b == pb)].groupby("it").ami.mean()
                if len(s):
                    ax.plot(s.index, s.values, "-o", ms=3, lw=1.6, color=PAIRCOL[(pa, pb)],
                            label=f"{PLAYER[pa]} vs {PLAYER[pb]}")
            sd = d[(d.kind == "seeds") & (d.proc == proc) & (d.player_a == "serial")].groupby("it").ami.mean()
            if len(sd):
                ax.plot(sd.index, sd.values, "--", lw=1.4, color="#52514e", label="bblean serial vs itself, other seed")
            if i == 0:
                ax.set_title(NAMES[ds])
            if j == 0:
                ax.set_ylabel(f"{PROC[proc]}\nAMI")
            if i == len(procs) - 1:
                ax.set_xlabel("refinement iteration")
            ax.set_xticks(range(0, C.PROCS[proc] + 1, 2 if C.PROCS[proc] > 5 else 1))
            ax.set_ylim(0, 0.85)
            ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Agreement between BitBIRCH players during bblean refinement (k = 3.5)", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_clusters(q, a, dss, path):
    """Cluster count by iteration (seed 0) and iteration-to-iteration AMI, recluster +0 and split + refit."""
    procs = ["rc0", "ref"]
    rows = [(p, "clusters") for p in procs] + [(p, "previous") for p in procs]
    fig, axs = plt.subplots(len(rows), len(dss), figsize=(2.6 * len(dss), 2.0 * len(rows)), squeeze=False)
    for j, ds in enumerate(dss):
        for i, (proc, what) in enumerate(rows):
            ax = axs[i, j]
            for pl in C.PLAYERS:
                if what == "clusters":
                    s = q[(q.dataset == ds) & (q.seed == 0) & (q.proc == proc) & (q.player == pl)].set_index("it").clusters
                    s = s / 1e3
                else:
                    s = a[(a.dataset == ds) & (a.seed == 0) & (a.kind == "previous") & (a.proc == proc)
                          & (a.player_a == pl)].set_index("it").ami
                if len(s):
                    ax.plot(s.index, s.values, "-o", ms=3, lw=1.6, color=PCOL[pl], label=PLAYER[pl])
            if what == "clusters":
                ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(NAMES[ds])
            if j == 0:
                ax.set_ylabel(f"{PROC[proc]}\n" + ("clusters (thousands)" if what == "clusters" else "AMI vs previous"))
            if i == len(rows) - 1:
                ax.set_xlabel("refinement iteration")
            ax.set_xticks(range(0, C.PROCS[proc] + 1, 2 if C.PROCS[proc] > 5 else 1))
            ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Cluster count and iteration-to-iteration change under bblean refinement (k = 3.5, seed 0)", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_quality(q, dss, path):
    """Quality before and after each procedure's last iteration, seed 0, vertical grouped bars."""
    metrics = [("clusters", "clusters (thousands)", 1e-3), ("isim", "within-cluster iSIM", 1),
               ("scaffold_purity", "scaffold purity", 1), ("scaf_ami", "AMI vs Murcko scaffolds", 1)]
    stages = [("none", 0, "unrefined")] + [(p, n, f"{PROC[p]} ×{n}") for p, n in C.PROCS.items()]
    fig, axs = plt.subplots(len(metrics), len(dss), figsize=(2.6 * len(dss), 2.0 * len(metrics)), squeeze=False)
    w = 0.26
    for j, ds in enumerate(dss):
        d = q[(q.dataset == ds) & (q.seed == 0)]
        for i, (m, lab, sc) in enumerate(metrics):
            ax = axs[i, j]
            for k, pl in enumerate(C.PLAYERS):
                vals = []
                for proc, it, _ in stages:
                    r = d[(d.player == pl) & (d.it == it) & ((d.proc == proc) if it else True)]
                    vals.append(r[m].iloc[0] * sc if len(r) else np.nan)
                x = np.arange(len(stages)) + (k - 1) * w
                ok = ~np.isnan(vals)
                ax.bar(x[ok], np.array(vals)[ok], w * 0.92, color=PCOL[pl], label=PLAYER[pl])
            ax.set_xticks(range(len(stages)))
            ax.set_xticklabels(["none", "rc +0\n×10", "rc +1σ\n×5", "split\n×5"] if i == len(metrics) - 1 else [])
            if i == 0:
                ax.set_title(NAMES[ds])
            if j == 0:
                ax.set_ylabel(lab)
            ax.grid(axis="y", color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Cluster quality before and after bblean refinement, per player (k = 3.5, seed 0)", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--paths"); ap.add_argument("--datasets", default="c35,np,scu,ena1m,chembl34")
    a = ap.parse_args(); cfg = C.load_cfg(a.paths)
    figs = Path(cfg["fig_dir"]); figs.mkdir(parents=True, exist_ok=True)
    ag, q = load(cfg, a.datasets.split(","))
    dss = [d for d in a.datasets.split(",") if d in set(q.dataset)]
    fig_players(ag, dss, figs / "agreement_between_players.png")
    fig_clusters(q, ag, dss, figs / "clusters_and_change_by_iteration.png")
    fig_quality(q, dss, figs / "quality_before_after.png")
    print("wrote", *sorted(p.name for p in figs.glob("*.png")))


if __name__ == "__main__":
    main()
