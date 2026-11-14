"""Inputs for the timing runs: each set in the seed-0 order, written once, untimed.

   python scripts/run/perf_inputs.py SET [SET ...] [--parts P ...]

Sets (perf.SETS): ena1m, chembl34, ena10m (full Enamine REAL 10.35M), zinc10m / zinc25m / zinc50m / zinc101m (the
first N rows of ZINC20 101M in the seed-0 order, so the smaller ZINC sets are random subsets of the big one).
Writes BB_OUT/perf_inputs/<base>_s0.npy (packed uint8, shuffled), and with --parts the `bb multiround` input dirs
BB_OUT/perf_inputs/<set>_parts<P>/partNNNN.npy (P equal pieces, file order = row order).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.run import perf  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sets", nargs="+")
ap.add_argument("--parts", nargs="*", type=int, default=[])
a = ap.parse_args()
for s in a.sets:
    base = perf.shuffled_path(s)
    if not base.exists():
        src, n_src, _ = perf.SETS[perf.base_of(s)]
        fps = np.load(src, mmap_mode="r")
        assert len(fps) == n_src, (s, len(fps))
        order = np.random.default_rng(0).permutation(n_src)
        out = np.lib.format.open_memmap(base.with_suffix(".partial.npy"), "w+", np.uint8, fps.shape)
        step = 1 << 22
        for i in range(0, n_src, step):  # sorted reads inside each block, written back in shuffled order
            idx = order[i:i + step]
            srt = np.argsort(idx)
            block = np.empty((len(idx), fps.shape[1]), np.uint8)
            block[srt] = fps[idx[srt]]
            out[i:i + step] = block
        out.flush(); del out
        base.with_suffix(".partial.npy").rename(base)
        print("wrote", base, flush=True)
    x = np.load(base, mmap_mode="r")[: perf.SETS_N[s]]
    for P in a.parts:
        d = perf.parts_dir(s, P)
        if d.exists():
            continue
        tmp = d.with_name(d.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=True)
        bounds = np.r_[0, np.cumsum([len(x) // P + (i < len(x) % P) for i in range(P)])]  # = np.array_split
        for i in range(P):
            np.save(tmp / f"part{i:04d}.npy", np.ascontiguousarray(x[bounds[i]:bounds[i + 1]]))
        tmp.rename(d)
        print("wrote", d, flush=True)
