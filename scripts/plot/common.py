"""Shared helpers for every plot script: implementations, data loading, style, saving.

Data is read only through scripts/common/paths.py (config/). Each loader returns what exists now, so a plot script
picks up new CSVs when they appear and draws only finished runs.

Implementations go by the names of scripts/common/implementations.py everywhere (tables, styles, code):
bblean_serial, bblean_split_merge (P = 36 in the comparison), bblean_batched (multithreaded CPU), nvmolkit_batched
(GPU). A results folder may also hold data/quality/by_run_earlier_labels/ (metrics of labels from an earlier run with
the same partitions); load_metrics reads it where it exists.
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import MAIN_PROCESSES as MAIN_P  # noqa: E402

RESULTS = P.RESULTS
PLOTS = P.PLOTS

DS_NAME = {"chembl34": "ChEMBL34", "scu": "SCUBIDOO", "ena1m": "Enamine 1M", "c35": "ChEMBL35 additions",
           "np": "ChEMBL33 NP"}
DS_SHORT = {"chembl34": "ChEMBL34", "scu": "SCUBIDOO", "ena1m": "Enamine\n1M", "c35": "ChEMBL35\nadditions",
            "np": "ChEMBL33\nNP"}
DS_ORDER = ["chembl34", "scu", "ena1m", "c35", "np"]

# House colours: colour = method.
SURFACE, INK, INK2, RULE = "#ffffff", "#0b0b0b", "#52514e", "#d6d5cf"
BB_SER, BB_PAR = "#eb6834", "#f4a582"           # bblean serial orange, bblean split-merge light orange
BB_SER_REF, BB_PAR_REF = "#a8400f", "#c48a6c"   # refined = darker shade of the same method
NV, NV_REF = "#76B900", "#4a7a00"               # nvMolKit green, refined dark green
OURS = "#1f3a68"                                # bblean batched (multithreaded CPU): dark blue
NEUTRAL, NEUTRAL2, NEUTRAL3 = "#4b6a88", "#9a9893", "#2f2e2b"  # non-method series (datasets, references)
PAR_NAME = "bblean split-merge"  # bblean's `bb multiround`
OURS_NAME = "bblean batched"  # nvMolKit's batched algorithm in bblean, multithreaded CPU
GPU_NAME = "nvMolKit batched (GPU)"
QP = {  # quality figures: style per implementation
    "bblean_serial": dict(label="bblean serial", color=BB_SER, ref=BB_SER_REF, marker="o", hollow=True),
    "bblean_split_merge": dict(label=PAR_NAME, color=BB_PAR, ref=BB_PAR_REF, marker="o", hollow=False),
    "nvmolkit_batched": dict(label=GPU_NAME, color=NV, ref=NV_REF, marker="o", hollow=False),
    "bblean_batched": dict(label=OURS_NAME, color=OURS, ref=OURS, marker="o", hollow=False),
}
PP = {  # timing figures: style per implementation
    "bblean_serial": dict(label="bblean serial", color=BB_SER, marker="o"),
    "bblean_split_merge": dict(label=PAR_NAME, color=BB_PAR, marker="o"),
    "nvmolkit_batched": dict(label=GPU_NAME, color=NV, marker="o"),
    "bblean_batched": dict(label=OURS_NAME, color=OURS, marker="o"),
}

COLW, FULLW = 3.4, 7.0  # inches: narrow / wide figure width


def style():
    """House style: white, light rules, grey ticks, bold left-aligned panel titles."""
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.labelsize": 7.5, "axes.labelcolor": INK2, "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": RULE, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    })


def plain_log(ax, axis="y", ticks=(0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)):
    """Log axis with plain-number tick labels (1,000 not 10^3) at the given ticks inside the current limits."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    a.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.10g}"))
    a.set_minor_locator(NullLocator()); a.set_minor_formatter(NullFormatter())


def fmt_s(v):
    """Seconds as a short plain number: 1,234 s, 98.7 s, 1.2 s."""
    return f"{v:,.0f} s" if v >= 100 else (f"{v:.1f} s" if v >= 1 else f"{v:.2f} s")


def end_labels(ax, items, gap=8, fontsize=6.5, dx=4, floor=None):
    """Direct labels [(x, y, text, colour)] just right of (x, y), nudged vertically (points) so they never overlap
    (from the earlier figures' common.end_labels). floor: keep every label at least this many points above the
    bottom of the axes."""
    if not items:
        return
    ax.figure.canvas.draw()
    k = 72 / ax.figure.dpi
    pts = sorted((ax.transData.transform((x, y))[1] * k, i) for i, (x, y, _, _) in enumerate(items))
    placed = [[yp, i] for yp, i in pts]
    for j in range(1, len(placed)):
        placed[j][0] = max(placed[j][0], placed[j - 1][0] + gap)
    shift = (sum(p[0] for p in placed) - sum(p[0] for p in pts)) / len(placed)
    for p in placed:
        p[0] -= shift
    for j in range(len(placed) - 2, -1, -1):
        placed[j][0] = min(placed[j][0], placed[j + 1][0] - gap)
    if floor is not None:
        lift = ax.transAxes.transform((0, 0))[1] * k + floor - placed[0][0]
        for p in placed:
            p[0] += max(lift, 0)
    orig = {i: yp for yp, i in pts}
    for yp, i in placed:
        x, y, text, color = items[i]
        ax.annotate(text, (x, y), xytext=(dx, yp - orig[i]), textcoords="offset points", color=color, va="center",
                    ha="left", fontsize=fontsize, fontweight="bold", annotation_clip=False)


def mean_ci(v):
    """Mean and 95 % t-interval of the values (whisker of the strip plots)."""
    from scipy.stats import t as tdist
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    m = v.mean()
    if len(v) < 2:
        return m, m, m
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def strip(ax, x, values, color, hollow=False, width=0.3, s=10, rng=None, mean=True):
    """One dot per input order (jittered), black mean line and 95 % CI whisker: the earlier quality figures' form."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v):
        return np.nan
    rng = rng if rng is not None else np.random.default_rng(0)
    xs = x + rng.uniform(-width * 0.3, width * 0.3, len(v))
    ax.scatter(xs, v, s=s, facecolor="none" if hollow else color, edgecolor=color, linewidth=1.1 if hollow else 0.6,
               zorder=3)
    m, lo, hi = mean_ci(v)
    if mean:
        if len(v) > 1:
            ax.plot([x, x], [lo, hi], color=INK, lw=0.6, zorder=2)
        ax.plot([x - width / 2, x + width / 2], [m, m], color=INK, lw=1.2, zorder=4)
    return m


def mean_point(ax, x, values, color, hollow=False, ms=5):
    """Mean as one point with its 95 % t-interval as an error bar (no per-run dots), clipped at 0 for non-negative
    data. Returns the mean."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v):
        return np.nan
    m, lo, hi = mean_ci(v)
    if (v >= 0).all():
        lo = max(lo, 0.0)  # a non-negative quantity (fraction, count): the interval is not drawn below zero
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]] if len(v) > 1 else None, fmt="o", color=color, ms=ms,
                mfc="white" if hollow else color, mew=1.2, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
    return m


def color_ticks(ax, keys, styles, axis="x"):
    """Colour each category tick label with its method colour, bold."""
    for tick, k in zip(ax.get_xticklabels() if axis == "x" else ax.get_yticklabels(), keys):
        tick.set_color(styles[k]["color"]); tick.set_fontweight("bold")


# Where each figure goes: plots/<topic>/<level>/<name>.png in the results folder. topic = quality, timing or
# other; level = primary (the headline figures), secondary (the next level of detail), extra (everything else, e.g.
# one figure per library). Each entry: (name pattern, topic/level, what it shows, data it uses); <ds> is a library
# key, <set> a MoleculeACE set, <k> the threshold multiplier, <proc> a refinement procedure. make_all.py writes
# plots/INDEX.md from this list.
_Q, _T, _R, _M, _Z = ("data/quality/by_run, by_run_earlier_labels", "data/timing/timings.csv",
                      "data/refinement/quality_<ds>.csv", "data/moleculeace/moleculeace_indices.csv",
                      "data/zinc22")
_A = "data/quality/agreement.csv"
FIGURES = [
    ("quality_chembl34", "quality/primary",
     "Cluster quality against chemistry (iSIM, scaffold purity, singletons, AMI and ARI with Murcko scaffolds and "
     "ChEMBL targets) of the four implementations on ChEMBL34, k = 3.5: mean and 95 % interval over input orders", _Q),
    ("agreement_chembl34", "quality/secondary",
     "Agreement (AMI, ARI) between runs on different input orders, ChEMBL34, k = 3.5: each implementation against "
     "itself and against bblean serial, next to the floor (bblean serial against itself); all molecules and the 20 "
     "and 50 largest clusters; mean and 95 % interval over pairs of orders", _A),
    ("agreement_by_library", "quality/secondary",
     "Agreement between runs (as agreement_chembl34) for every library, k = 3.5", _A),
    ("agreement_vs_k", "quality/secondary",
     "Agreement between runs against k (3, 3.5, 4) for every library, means joined along k", _A),
    ("quality_by_library", "quality/secondary", "Quality against chemistry (as quality_chembl34) for every library", _Q),
    ("quality_vs_k", "quality/secondary", "Quality against k for every library (mean and 95 % interval)", _Q),
    ("refinement_chembl34", "quality/secondary",
     "Quality on ChEMBL34 before and after 10 passes of bblean's recluster refinement",
     "data/refinement/quality_chembl34.csv"),
    ("refinement_passes_all", "quality/secondary",
     "Quality against the refinement pass (recluster, no threshold increase) for every library, all input orders", _R),
    ("quality_vs_k_<ds>", "quality/extra", "Quality against k for one library, one dot per input order", _Q),
    ("quality_all_scores", "quality/extra",
     "AMI, ARI, homogeneity, completeness and V-measure against scaffolds and targets, every library", _Q),
    ("clusters_by_library", "quality/extra", "Number of clusters per implementation and library, k = 3.5", _Q),
    ("ami_vs_clusters_<ds>", "quality/extra",
     "AMI with scaffolds and targets against the number of clusters (k = 3, 3.5, 4), one library", _Q),
    ("topk_<ds>", "quality/extra", "Quality of the largest clusters (5, 20, 100, 1000 largest), one library",
     "data/quality/by_run/topk_<ds>.csv"),
    ("top20_clusters_<ds>_k<k>", "quality/extra",
     "Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k",
     "data/quality/top_clusters.csv, size_bins.csv"),
    ("refinement_passes_rc0_<ds>", "quality/extra",
     "Quality against the refinement pass (recluster, no threshold increase), one library, input order 0", _R),
    ("refinement_passes_rc1_all", "quality/extra",
     "Quality against the refinement pass (recluster, threshold grows by sigma each pass), every library", _R),
    ("refinement_passes_rc1_<ds>", "quality/extra",
     "Quality against the refinement pass (recluster, threshold grows by sigma each pass), one library", _R),
    ("refinement_passes_ref_all", "quality/extra",
     "Quality against the refinement pass (refine the largest cluster), every library", _R),
    ("refinement_passes_ref_<ds>", "quality/extra",
     "Quality against the refinement pass (refine the largest cluster), one library", _R),
    ("time_vs_size_zinc20", "timing/primary", "Clustering time against size, ZINC20 subsets of 10M to 101M molecules", _T),
    ("time_vs_workers_enamine10m", "timing/primary",
     "Clustering time against CPU threads / processes on Enamine 10M, linear and log-log axes", _T),
    ("time_vs_workers_chembl34", "timing/secondary", "Clustering time against CPU threads / processes on ChEMBL34", _T),
    ("memory_zinc20", "timing/secondary", "Peak memory against size, ZINC20 subsets", _T),
    ("zinc22_cluster_sizes", "other/primary",
     "Sizes of the 10,000 largest clusters of the 12.16 billion molecule ZINC22 run (nvMolKit batched)",
     "data/zinc22/largest_cluster_sizes.csv"),
    ("alg_schematic", "other/secondary", "Schematic of one round of the batched algorithm (no data)", "none"),
    ("moleculeace_by_set", "other/secondary",
     "MoleculeACE, threshold 0.65: clusters and three clustering indices, mean over the 30 sets", _M),
    ("moleculeace_clusters_vs_threshold_<set>", "other/extra",
     "Clusters with at least 1, 2, 3, 5, 10 members against threshold, one MoleculeACE set", _M),
    ("moleculeace_indices_vs_threshold_<set>", "other/extra",
     "Calinski-Harabasz, Davies-Bouldin and Dunn index against threshold, one MoleculeACE set", _M),
    ("bbl_serial_vs_parallel_top20_<ds>", "other/extra",
     "Sizes and iSIM of the 20 largest clusters, bblean serial against the others, one library",
     "data/quality/top_clusters.csv"),
    ("zinc22_vs_libraries", "other/extra",
     "The ZINC22 run next to three smaller libraries (nvMolKit batched): whole clustering and largest clusters",
     f"{_Z}/zinc22_overall.csv, zinc22_topk.csv; {_Q}"),
]


def _regex(pattern):
    """Name pattern -> regular expression: <ds>, <set>, <k>, <proc> match one name part each."""
    rx = re.escape(pattern)
    for tag in ("ds", "set", "proc"):
        rx = rx.replace(re.escape(f"<{tag}>"), r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)??" if tag == "set" else r"[A-Za-z0-9]+")
    rx = rx.replace(re.escape("<k>"), r"[0-9.]+")
    return re.compile(rx + "$")


def placement(name):
    """(topic/level folder, FIGURES entry) of a figure name; the first matching entry wins."""
    for entry in FIGURES:
        if _regex(entry[0]).match(name):
            return entry[1], entry
    raise KeyError(f"figure {name!r} has no entry in FIGURES (scripts/plot/common.py)")


def save(fig, name):
    """Write plots/<topic>/<level>/<name>.png in the results folder (placement from FIGURES)."""
    outdir = PLOTS / placement(name)[0]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def waiting(name, what):
    print(f"WAITING {name}: {what}")


# ---------------------------------------------------------------- quality metrics (earlier + new)

def _metric_files(ds, topk):
    name = f"topk_{ds}.csv" if topk else f"{ds}.csv"
    # by_run_earlier_labels: metrics of labels from an earlier run with the same partitions; by_run: written by
    # scripts/metrics/metrics.py
    return [P.QUALITY / "by_run_earlier_labels" / name, P.QUALITY / "by_run" / name]


def any_metrics(topk=False):
    """True if any per-run metrics CSV exists for any library (else the quality figures wait)."""
    return any(f.exists() for ds in DS_ORDER for f in _metric_files(ds, topk))


def load_metrics(ds, topk=False):
    """Rows of the metrics CSVs for dataset ds, one per implementation/k/seed(/k_top); bblean split-merge at MAIN_P
    processes only."""
    frames = [pd.read_csv(f, dtype={"processes": "Int64"}).assign(_earlier=i == 0)
              for i, f in enumerate(_metric_files(ds, topk)) if f.exists()]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d.processes.isna() | (d.processes == MAIN_P)]
    # on duplicates (same partitions) keep: bblean serial from by_run, the others from the earlier labels; this order
    # also fixes the order of the runs, which sets the jitter of the strip plots
    d["_pri"] = np.where(~d._earlier & (d.implementation == "bblean_serial"), 0, np.where(d._earlier, 1, 2))
    key = ["implementation", "bf", "k", "seed"] + (["k_top"] if topk else [])
    d = d.sort_values("_pri", kind="stable").drop_duplicates(key, keep="first")
    return d.drop(columns=["_pri", "_earlier"]).reset_index(drop=True)


def implementations_present(d, order=("bblean_serial", "bblean_split_merge", "bblean_batched", "nvmolkit_batched")):
    return [p for p in order if p in set(d.get("implementation", []))]


def agg_seeds(d, cols, by=("implementation", "bf", "k")):
    """Mean, min and max over seeds."""
    g = d.groupby(list(by))[cols]
    out = g.mean().add_suffix("_mean").join(g.min().add_suffix("_min")).join(g.max().add_suffix("_max"))
    out = out.join(d.groupby(list(by)).size().rename("n_seeds"))
    return out.reset_index()


def errbar(ax, x, mean, lo, hi, p, impls=QP, line=True, **kw):
    st = impls[p]
    yerr = np.vstack([np.asarray(mean) - np.asarray(lo), np.asarray(hi) - np.asarray(mean)])
    if st.get("hollow"):
        kw.setdefault("markerfacecolor", SURFACE)
    if p in OVERLAY:  # bblean batched ring under the nvMolKit dot (identical clusters)
        scale = kw.pop("ms", kw.pop("markersize", plt.rcParams["lines.markersize"])) / 4
        kw = {**overlay(p, scale), **kw}
    ax.errorbar(x, mean, yerr=yerr, color=st["color"], marker=st["marker"], label=st["label"],
                linestyle="-" if line else "none", capsize=1.5, elinewidth=0.8, **kw)


def legend_below(fig, axes, ncol=3, y=0.0):
    hs, ls = [], []
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in ls:
                hs.append(h); ls.append(lab)
    fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


# ---------------------------------------------------------------- timing CSVs (data/timing/)

PERF_DS = {"ena1m": "Enamine 1M", "chembl34": "ChEMBL34", "ena10m": "Enamine 10M", "zinc10m": "ZINC20 10M",
           "zinc25m": "ZINC20 25M", "zinc50m": "ZINC20 50M", "zinc101m": "ZINC20 101M"}


def load_perf():
    """Rows of data/timing/timings.csv (scripts/run/perf_run.py writes it) as: implementation, dataset, n, threads
    (threads for bblean batched, processes for split-merge), seconds (best of repeats), peak_gib (GPU pool for nvMolKit,
    host peak RSS otherwise), gpu_input_gib. k 3.5 and bf 1024 only. Rows with `superseded` = True are skipped."""
    f = P.TIMING / "timings.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    out = pd.DataFrame({"implementation": d["implementation"], "dataset": d["set"].astype(str),
                        "n": d["n"], "threads": d["threads"].fillna(d["processes"]), "seconds": d["seconds_min"]})
    gpu_row = out["implementation"].eq("nvmolkit_batched")
    out["gpu_input_gib"] = d["gpu_input_gib"]
    out["peak_gib"] = d["gpu_pool_peak_gib"].where(gpu_row, d["host_peak_gib"])
    out["bf"], out["k"], out["when"] = d["bf"], d["k"], d["when"]
    ok = ~d["superseded"].astype(str).str.lower().eq("true") if "superseded" in d else True
    out = out[ok & out.implementation.isin(list(PP)) & out.seconds.notna() & (out.k == 3.5) & (out.bf == P.BF)]
    key = ["implementation", "dataset", "n", "threads", "bf"]
    # one row per setting: the latest run (each row is already the best of its repeats)
    out = out.sort_values("when").drop_duplicates(key, keep="last")
    return out[key + ["seconds", "peak_gib", "gpu_input_gib"]].reset_index(drop=True)


def main_setting(perf):
    """One row per implementation and size at the main setting: bblean serial 1 thread, split-merge and bblean batched
    at MAIN_P (36) processes/threads (or their only setting), nvMolKit as measured."""
    rows = []
    for (p, ds, n), g in perf.groupby(["implementation", "dataset", "n"]):
        if p in ("bblean_split_merge", "bblean_batched") and g.threads.notna().any():
            g = g[g.threads == MAIN_P] if (g.threads == MAIN_P).any() else g.iloc[:0]
        if len(g):
            rows.append(g.sort_values("seconds").iloc[0])
    return pd.DataFrame(rows)


def logx_plain(ax):
    """Log x axis with plain number tick labels at 1, 2, 5 x 10^n (no 3x10^1 style labels)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(MaxNLocator(4) if hi / lo < 4 else LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def refine_mr_p():
    """Process count of the split-merge runs in the refinement experiment (36 parts, 36 processes). Override with
    BB_REFINE_MR_P."""
    return int(P.CFG.get("BB_REFINE_MR_P", 36))


FAMILY = [("zinc", "ZINC20", "-"), ("ena", "Enamine REAL", "--"), ("chembl", "ChEMBL34", ":")]


def family(ds):
    for pre, name, ls in FAMILY:
        if str(ds).startswith(pre):
            return name, ls
    return str(ds), "-."


def plot_vs_n(ax, m, col, impls=None):
    """col against N, one colour per implementation; sizes of one data family joined by a line (style = family), since
    subsets of one library are one knob and different libraries are not."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    used = {}
    for p in impls or PP:
        s = m[(m.implementation == p)].dropna(subset=[col])
        for fam, g in s.groupby(s.dataset.map(lambda x: family(x)[0])):
            ls = family(g.dataset.iloc[0])[1]
            g = g.sort_values("n")
            ax.plot(g.n, g[col], ls=ls if len(g) > 1 else "none", marker=PP[p]["marker"], color=PP[p]["color"])
            if len(g) > 1:
                used[fam] = ls
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("molecules")
    if len(used) < 2:
        used = {}
    handles = [Line2D([], [], color=PP[p]["color"], marker=PP[p]["marker"], ls="none", label=PP[p]["label"])
               for p in (impls or PP) if (m.implementation == p).any()]
    handles += [Line2D([], [], color="0.4", ls=ls, label=fam) for fam, ls in used.items()]
    return handles


# ---------------------------------------------------------------- bblean batched and nvMolKit overlap
# bblean batched and nvMolKit batched give identical clusters, so their points coincide. bblean batched is drawn as a
# larger open ring under a smaller filled nvMolKit dot, so both stay visible. Use with ax.plot(..., marker="o", **overlay("bblean_batched")).
OVERLAY = {"bblean_batched": dict(markersize=6.0, markerfacecolor="none", markeredgewidth=1.1, zorder=3),
           "nvmolkit_batched": dict(markersize=2.8, markeredgewidth=0, zorder=4)}


def overlay(p, ms_scale=1.0):
    """Marker kwargs for ax.plot / errorbar: ring for bblean batched, small dot for nvMolKit, {} for the others."""
    kw = dict(OVERLAY.get(p, {}))
    if "markersize" in kw:
        kw["markersize"] *= ms_scale
    return kw


def panel_label(p, impls=None):
    """Implementation name wrapped onto two lines for a narrow panel title ("bblean parallel:\nsplit-merge")."""
    lab = (impls or QP)[p]["label"]
    return lab.replace(": ", ":\n").replace(" (GPU)", "\n(GPU)")
