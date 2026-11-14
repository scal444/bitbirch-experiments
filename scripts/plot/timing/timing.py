"""Speed figures: time against size and time against workers for the four implementations.

  (a) time_vs_size_zinc20: time against size, ZINC20 subsets 10M-101M (log-log; bblean serial on 1
      thread, bblean split-merge and bblean batched at 36 processes / threads, nvMolKit batched on the GPU), direct
      labels with the time at the largest size and nvMolKit's speed-up over bblean serial there;
  (b) time_vs_workers_enamine10m: time against workers on Enamine 10M, linear axes beside log-log axes:
      bblean batched against threads, bblean split-merge (parts = processes = P) against processes, bblean serial
      as one point at 1, nvMolKit as a dashed line. The same pair for ChEMBL34: time_vs_workers_chembl34
  Peak memory against size (host peak RSS for CPU implementations, GPU pool + input for nvMolKit):
      memory_zinc20
k = 3.5, bf 1024. Only measured runs are drawn. Data: data/timing/timings.csv (scripts/run/run_timings.sh).
Output: plots/timing/ of the results folder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import common as C  # noqa: E402

SIZE_PLAYERS = ["serial", "multiround", "cpu_batched", "gpu"]
MAIN_SWEEP = "ena10m"  # the ChEMBL34 sweep is a secondary figure
SI_SWEEP = "chembl34"


def threshold(s):
    from scripts.run import perf
    return perf.SETS[perf.base_of(s)][2]


def size_panel(ax, m, col, unit):
    items = []
    last = {}
    for p in SIZE_PLAYERS:
        g = m[m.player == p].dropna(subset=[col]).sort_values("n")
        if g.empty:
            continue
        st = C.PP[p]
        ax.plot(g.n / 1e6, g[col], color=st["color"], marker="o", ms=3)
        x, y = g.n.iloc[-1] / 1e6, g[col].iloc[-1]
        last[p] = (x, y)
        val = C.fmt_s(y) if unit == "s" else f"{y:,.0f} GiB" if y >= 10 else f"{y:.1f} GiB"
        items.append((x, y, f"{st['label']}  {val}", st["color"]))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8, 1800)
    ax.set_xticks([10, 25, 50, 101.27], ["10", "25", "50", "101"]); ax.minorticks_off()
    ax.set_xlabel("molecules (millions)")
    return items, last


def workers_panel(ax, perf, ds, loglog=False):
    g_b = perf[(perf.player == "cpu_batched") & (perf.dataset == ds)].dropna(subset=["threads"]).sort_values("threads")
    g_m = perf[(perf.player == "multiround") & (perf.dataset == ds)].dropna(subset=["threads"]).sort_values("threads")
    ser = perf[(perf.player == "serial") & (perf.dataset == ds)]
    gpu = perf[(perf.player == "gpu") & (perf.dataset == ds)]
    items = []
    for g, p in ((g_m, "multiround"), (g_b, "cpu_batched")):
        if len(g):
            st = C.PP[p]
            ax.plot(g.threads, g.seconds, color=st["color"], marker="o", ms=3)
            items.append((g.threads.iloc[-1], g.seconds.iloc[-1], f"{st['label']}  {C.fmt_s(g.seconds.iloc[-1])}",
                          st["color"]))
    if len(gpu):
        v = gpu.seconds.min()
        ax.plot([1, 72], [v, v], color=C.NV, lw=1.2, ls="--")
        items.append((72, v, f"{C.PP['gpu']['label']}  {C.fmt_s(v)}", C.NV))
    if len(ser):
        v = ser.seconds.min()
        ring = dict(mfc="none", mew=1.2, ms=7, zorder=2)
        ax.plot([1], [v], marker="o", color=C.BB_SER, ls="none", **ring)  # ring: the 1-worker points stay visible
        ax.annotate(f"{C.PP['serial']['label']}  {C.fmt_s(v)}", (1, v), xytext=(12, 12), textcoords="offset points",
                    color=C.BB_SER, fontsize=6.5, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=C.BB_SER, lw=0.6, shrinkA=0, shrinkB=3))
    ax.set_xlabel("CPU threads / processes")
    if loglog:
        ax.set_xscale("log"); ax.set_xlim(0.8, 400)
        C.plain_log(ax, axis="x", ticks=(1, 2, 4, 8, 16, 36, 72))
        ax.set_yscale("log")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 2.2)
        C.plain_log(ax)
        ax.set_title(f"{C.PERF_DS[ds]}, log-log axes")
    else:
        ax.set_xlim(-1, 120)
        ax.set_xticks([1, 16, 36, 72])
        ax.set_ylim(0, None)
        ax.set_title(f"{C.PERF_DS[ds]}, linear axes")
    return items


def main():
    C.style()
    perf = C.load_perf()
    zinc = perf[perf.dataset.astype(str).str.startswith("zinc")] if len(perf) else perf
    if zinc.empty:
        C.waiting("time_vs_size_zinc20", "ZINC20 timings in data/timing/timings.csv"); return
    m = C.main_setting(zinc.dropna(subset=["n"]))
    fig, ax = plt.subplots(figsize=(C.COLW, 2.6))  # time against size
    items, last = size_panel(ax, m, "seconds", "s")
    ax.set_xlim(8, 1500)
    ax.set_ylim(1, 5000); C.plain_log(ax)
    ax.set_ylabel("seconds"); ax.set_title("ZINC20 subsets")
    C.end_labels(ax, items, fontsize=6)
    fig.tight_layout()
    C.save(fig, "time_vs_size_zinc20")

    for ds, name in ((MAIN_SWEEP, "time_vs_workers_enamine10m"),  # time against threads / processes
                     (SI_SWEEP, "time_vs_workers_chembl34")):
        fig, axs = plt.subplots(1, 2, figsize=(C.FULLW, 2.6))
        for ax, loglog in zip(axs, (False, True)):
            C.end_labels(ax, workers_panel(ax, perf, ds, loglog=loglog), floor=5)
        axs[0].set_ylabel("seconds")
        fig.tight_layout(w_pad=1.5)
        C.save(fig, name)

    fig, ax = plt.subplots(figsize=(C.COLW, 2.6))  # memory against size
    mm = m.copy()  # nvMolKit: GPU pool peak plus the device copy of the input, as host peak RSS includes it for the CPU
    g = mm.player == "gpu"
    mm.loc[g, "peak_gib"] = mm.loc[g, "peak_gib"] + mm.loc[g, "gpu_input_gib"]
    items, _ = size_panel(ax, mm, "peak_gib", "GiB")
    ax.set_ylim(1, 100); C.plain_log(ax)
    ax.set_ylabel("peak memory (GiB)"); ax.set_title("Peak memory, ZINC20 subsets")
    C.end_labels(ax, items)
    fig.tight_layout()
    C.save(fig, "memory_zinc20")


if __name__ == "__main__":
    main()
