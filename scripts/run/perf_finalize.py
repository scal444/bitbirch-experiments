"""Fit scaling exponents from data/timing/timings.csv (results folder). Run after the last timing (run_timings.sh does).

   python scripts/run/perf_finalize.py [--write]     (without --write: print only)

data/timing/scaling_exponents.csv: per implementation (bblean_batched at 36 threads, bblean_split_merge at 36
processes, bblean_serial, nvmolkit_batched) the slope
b of log(seconds) = a + b log(N), fitted on (1) the ZINC20 sizes 10M-101M (one dataset, random subsets of each
other) and (2) all sizes (Enamine 1M, ChEMBL34, Enamine 10M, ZINC 10M-101M; mixed datasets), plus the same for
peak memory (host for the CPU implementations, GPU pool + input for nvMolKit). Rows with `superseded` = True are left
out (see the results folder README).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

d = pd.read_csv(P.TIMING / "timings.csv")
WRITE = "--write" in sys.argv
u = d[~d["superseded"].astype(str).str.lower().eq("true")] if "superseded" in d else d
u = u.copy()  # batched sweep rows at 36 threads count as size points
u = u[(u.implementation != "bblean_batched") | (u.threads == 36)]
u = u[(u.implementation != "bblean_split_merge") | (u.processes == 36)]
u["mem_gib"] = np.where(u.implementation == "nvmolkit_batched",
                        u.gpu_pool_peak_gib.fillna(0) + u.gpu_input_gib.fillna(0),
                        u.host_peak_gib)
rows = []
for impl, g in u.groupby("implementation"):
    g = g.sort_values("when").drop_duplicates("set", keep="last")  # latest row per size
    for scope, sel in (("zinc_10m_101m", g.set.str.startswith("zinc")), ("all_sizes", np.ones(len(g), bool))):
        h = g[sel]
        if len(h) < 2:
            continue
        x = np.log(h.n.astype(float))
        rows.append(dict(implementation=impl, scope=scope, points=len(h), sizes=" ".join(h.sort_values("n").set),
                         time_exponent=np.polyfit(x, np.log(h.seconds_min.astype(float)), 1)[0],
                         memory_exponent=np.polyfit(x, np.log(h.mem_gib.astype(float)), 1)[0]))
if WRITE:
    pd.DataFrame(rows).to_csv(P.TIMING / "scaling_exponents.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
