"""Agreement between the cluster memberships of two runs (AMI, ARI), within and across implementations.

How the BitBIRCH authors judge an implementation: its agreement with bblean serial, set against the floor, the
agreement of bblean serial with itself when only the input order changes. Every pair uses two different input orders
(scripts/metrics/agreement.py). Seven comparisons, colour = implementation (the one that is not bblean serial),
square = the same implementation on two input orders, circle = against bblean serial; the floor is also drawn as a
dashed line at its mean. Mean and 95 % t-interval over all pairs.

Output (plots/quality/ of the results folder):
  agreement_chembl34    ChEMBL34, k = 3.5; columns = all molecules, 20 and 50 largest clusters; rows = AMI, ARI
  agreement_by_library  every library, k = 3.5; rows = AMI and ARI on each subset, columns = libraries
  agreement_vs_k        every library, k = 3, 3.5, 4; means joined along k
Data: data/quality/agreement.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import common as C  # noqa: E402

CSV = C.P.QUALITY / "agreement.csv"
# (implementation_a, implementation_b): same implementation first (floor first), then against bblean serial
SER, SM, BB, NV = "bblean_serial", "bblean_split_merge", "bblean_batched", "nvmolkit_batched"
COMPS = [(SER, SER), (SM, SM), (BB, BB), (NV, NV), (SER, SM), (SER, BB), (SER, NV)]
SUBSETS = [("all", "all molecules"), ("top20", "20 largest clusters"), ("top50", "50 largest clusters")]
SCORES = [("ami", "AMI"), ("ari", "ARI")]
XPOS = [0, 1, 2, 3, 4.7, 5.7, 6.7]  # a gap between the two groups (same implementation | against bblean serial)
K = 3.5
FLOOR = (SER, SER)


def style(c):
    a, b = c
    st = C.QP[b]
    return dict(color=st["color"], marker="s" if a == b else "o")


def label(c):
    a, b = c
    if c == FLOOR:
        return "bblean serial vs itself (floor)"
    if a == b:
        return f"{C.QP[a]['label'].replace(' (GPU)', '')} vs itself"
    return f"bblean serial vs {C.QP[b]['label'].replace(' (GPU)', '')}"


def values(d, ds, k, c, subset, score):
    s = d[(d.dataset == ds) & (d.k == k) & (d.implementation_a == c[0]) & (d.implementation_b == c[1])
          & (d.subset == subset)]
    return s[score].to_numpy()


def point(ax, x, v, c, ms=4.5):
    st = style(c)
    m, lo, hi = C.mean_ci(v)
    lo = max(lo, 0.0)
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]] if len(v) > 1 else None, fmt=st["marker"], color=st["color"],
                ms=ms, mew=1.0, elinewidth=1.0, capsize=2.0, capthick=1.0, zorder=3)
    return m


def comparisons_panel(ax, d, ds, subset, score, ms=4.5):
    """x = comparison; mean and 95 % interval over pairs; floor mean as a dashed line."""
    shown = []
    for x, c in zip(XPOS, COMPS):
        v = values(d, ds, K, c, subset, score)
        if len(v):
            m = point(ax, x, v, c, ms=ms)
            if c == FLOOR:
                ax.axhline(m, color=C.QP[SER]["color"], ls="--", lw=0.8, zorder=1)
            shown.append(c)
    ax.set_xlim(-0.6, XPOS[-1] + 0.6)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    return shown


def handles(comps=COMPS):
    hs = []
    for c in comps:
        st = style(c)
        hs.append(Line2D([], [], marker=st["marker"], color=st["color"], ms=4.5, ls="--" if c == FLOOR else "none",
                         lw=0.8, label=label(c)))
    return hs


def fig_chembl34(d):
    ds = "chembl34"
    fig, axs = plt.subplots(2, 3, figsize=(C.FULLW, 4.4), sharey=True)
    for r, (score, sname) in enumerate(SCORES):
        for c_, (subset, sub_name) in enumerate(SUBSETS):
            ax = axs[r, c_]
            comparisons_panel(ax, d, ds, subset, score)
            if r == 0:
                ax.set_title(sub_name, fontsize=7.5)
            if c_ == 0:
                ax.set_ylabel(sname)
    n = d[(d.dataset == ds) & (d.k == K) & (d.implementation_a == SER) & (d.implementation_b == SER)]
    n_orders = len(set(n.order_a) | set(n.order_b))
    fig.suptitle(f"Agreement between runs on different input orders, ChEMBL34, k = 3.5 ({n_orders} orders)",
                 x=0.01, ha="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.97), h_pad=0.8, w_pad=0.6)
    fig.legend(handles=handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=6.5,
               columnspacing=1.2)
    C.save(fig, "agreement_chembl34")


def fig_by_library(d):
    dss = [ds for ds in C.DS_ORDER if ((d.dataset == ds) & (d.k == K)).any()]
    rows = [(score, sname, subset, sub_name) for score, sname in SCORES for subset, sub_name in SUBSETS]
    fig, axs = plt.subplots(len(rows), len(dss), figsize=(C.FULLW, 1.05 * len(rows) + 0.9), sharey=True,
                            squeeze=False)
    for r, (score, sname, subset, sub_name) in enumerate(rows):
        for c_, ds in enumerate(dss):
            ax = axs[r, c_]
            comparisons_panel(ax, d, ds, subset, score, ms=3.5)
            ax.tick_params(labelsize=5.5)
            if r == 0:
                ax.set_title(C.DS_NAME[ds], fontsize=7)
            if c_ == 0:
                ax.set_ylabel(f"{sname},\n{sub_name.replace(' clusters', '')}", fontsize=6.5)
    fig.suptitle("Agreement between runs on different input orders, every library, k = 3.5", x=0.01, ha="left",
                 fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.085, 1, 0.98), h_pad=0.5, w_pad=0.4)
    fig.legend(handles=handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=6.3,
               columnspacing=1.2)
    C.save(fig, "agreement_by_library")


def fig_vs_k(d):
    dss = [ds for ds in C.DS_ORDER if (d.dataset == ds).any()]
    ks = sorted(set(d.k))
    rows = [(score, sname, subset, sub_name) for score, sname in SCORES for subset, sub_name in SUBSETS]
    fig, axs = plt.subplots(len(rows), len(dss), figsize=(C.FULLW, 1.05 * len(rows) + 0.9), sharey=True,
                            squeeze=False)
    w = 0.05
    for r, (score, sname, subset, sub_name) in enumerate(rows):
        for c_, ds in enumerate(dss):
            ax = axs[r, c_]
            for j, c in enumerate(COMPS):
                st = style(c)
                xs, ms = [], []
                for k in ks:
                    v = values(d, ds, k, c, subset, score)
                    if not len(v):
                        continue
                    x = k + (j - (len(COMPS) - 1) / 2) * w
                    m, lo, hi = C.mean_ci(v)
                    lo = max(lo, 0.0)
                    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=st["marker"], color=st["color"], ms=2.6,
                                mew=0.8, elinewidth=0.7, capsize=0, zorder=3)
                    xs.append(x); ms.append(m)
                if len(xs) > 1:
                    ax.plot(xs, ms, color=st["color"], lw=0.7, ls="--" if c == FLOOR else "-", zorder=2)
            ax.set_xticks(ks, [f"{k:g}" for k in ks] if r == len(rows) - 1 else [])
            ax.set_xlim(min(ks) - 0.3, max(ks) + 0.3)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=5.5)
            if r == len(rows) - 1:
                ax.set_xlabel("k", fontsize=6.5)
            if r == 0:
                ax.set_title(C.DS_NAME[ds], fontsize=7)
            if c_ == 0:
                ax.set_ylabel(f"{sname},\n{sub_name.replace(' clusters', '')}", fontsize=6.5)
    fig.suptitle("Agreement between runs on different input orders against k, every library", x=0.01, ha="left",
                 fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.085, 1, 0.98), h_pad=0.5, w_pad=0.4)
    fig.legend(handles=handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=6.3,
               columnspacing=1.2)
    C.save(fig, "agreement_vs_k")


def main():
    C.style()
    if not CSV.exists():
        for n in ("agreement_chembl34", "agreement_by_library", "agreement_vs_k"):
            C.waiting(n, "data/quality/agreement.csv (scripts/metrics/agreement.py)")
        return
    d = pd.read_csv(CSV)
    if ((d.dataset == "chembl34") & (d.k == K)).any():
        fig_chembl34(d)
    else:
        C.waiting("agreement_chembl34", "ChEMBL34 rows at k = 3.5 in data/quality/agreement.csv")
    fig_by_library(d)
    fig_vs_k(d)


if __name__ == "__main__":
    main()
