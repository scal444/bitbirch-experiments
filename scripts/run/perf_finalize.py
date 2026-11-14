"""Tidy data/timing/timings.csv (results folder) and fit scaling exponents. Run after the last timing (run_timings.sh does).

   python scripts/run/perf_finalize.py [--write]     (without --write: print only)

- Adds `superseded` (True/False): in the example results, a GPU row timed while other processes held the GPU (tag
  other_gpu_procs>=1) is superseded by a later rerun with other_gpu_procs=0. New runs don't write that tag, so their
  rows are never superseded. Rows are never deleted.
- data/timing/scaling_exponents.csv: per implementation (`player` column: batched = bblean batched at 36 threads,
  multiround = bblean split-merge at 36 processes, serial = bblean serial, nvmolkit = nvMolKit batched GPU) the slope
  b of log(seconds) = a + b log(N), fitted on (1) the ZINC20 sizes 10M-101M (one dataset, random subsets of each
  other) and (2) all sizes (Enamine 1M, ChEMBL34, Enamine 10M, ZINC 10M-101M; mixed datasets), plus the same for
  peak memory (host for the CPU implementations, GPU pool + input for nvMolKit).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

f = P.TIMING / "timings.csv"
d = pd.read_csv(f)
d["tag"] = d["tag"].fillna("")
d["other_gpu_procs"] = d.tag.map(lambda t: int(m.group(1)) if (m := re.search(r"other_gpu_procs=(\d+)", t)) else np.nan)
key = ["player", "set", "threads", "processes", "bf", "k"]
sup = np.zeros(len(d), bool)
for i, r in d.iterrows():
    if r.player == "nvmolkit" and r.other_gpu_procs >= 1:
        same = (d[key].fillna(-1) == r[key].fillna(-1)).all(axis=1) & (d.other_gpu_procs == 0) & (d.index > i)
        sup[i] = bool(same.any())
d["superseded"] = sup
WRITE = "--write" in sys.argv
if WRITE:
    d.drop(columns="other_gpu_procs").to_csv(f, index=False)

u = d[~d.superseded].copy()  # batched sweep rows at 36 threads count as size points
u = u[(u.player != "batched") | (u.threads == 36)]
u = u[(u.player != "multiround") | (u.processes == 36)]
u["mem_gib"] = np.where(u.player == "nvmolkit", u.gpu_pool_peak_gib.fillna(0) + u.gpu_input_gib.fillna(0),
                        u.host_peak_gib)
rows = []
for player, g in u.groupby("player"):
    g = g.sort_values("when").drop_duplicates("set", keep="last")  # latest row per size
    for scope, sel in (("zinc_10m_101m", g.set.str.startswith("zinc")), ("all_sizes", np.ones(len(g), bool))):
        h = g[sel]
        if len(h) < 2:
            continue
        x = np.log(h.n.astype(float))
        rows.append(dict(player=player, scope=scope, points=len(h), sizes=" ".join(h.sort_values("n").set),
                         time_exponent=np.polyfit(x, np.log(h.seconds_min.astype(float)), 1)[0],
                         memory_exponent=np.polyfit(x, np.log(h.mem_gib.astype(float)), 1)[0]))
if WRITE:
    pd.DataFrame(rows).to_csv(P.TIMING / "scaling_exponents.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
