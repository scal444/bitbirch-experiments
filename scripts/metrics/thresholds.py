"""Thresholds as in the authors' recipe (Lopez-Perez & Miranda-Quintana 2026): t = iSIM + k * iSIM-sigma.

  python scripts/metrics/thresholds.py [DS ...]      (default: the five quality datasets)

DS: chembl34 scu ena1m c35 np (fingerprints under BB_DATA, see scripts/common/paths.py), or the timing sets ena10m,
zinc101m (scripts/run/perf.py). Uses the bblean package installed in the environment (bblean.similarity).
Adds the computed libraries to data/thresholds/isim_sigma.json of the results folder (keeps the others), which
every other script reads.
"""
import json
import sys
import time
from pathlib import Path

import bblean.similarity as S
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.run import perf  # noqa: E402

FILES = {ds: d["fps"] for ds, d in P.DATASETS.items()}
FILES.update({s: perf.SETS[s][0] for s in ("ena10m", "zinc101m")})
out = {}
for name in sys.argv[1:] or list(P.DATASETS):
    fps = np.load(FILES[name])
    if fps.shape[1] == 2048: fps = np.packbits(fps, axis=1)
    t = time.time()
    isim = S.jt_isim_packed(fps)
    sig50 = S.estimate_jt_std(fps, n_samples=50)          # as in the authors' best_practices_functions.py
    sigdef = S.estimate_jt_std(fps)                          # library default heuristic
    r = dict(n=len(fps), isim=isim, sigma_n50=sig50, sigma_default=sigdef,
             **{f"t_{k}": isim + k * sig50 for k in (3, 3.5, 4)}, secs=time.time() - t)
    out[name] = r; print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
dest = P.THRESHOLDS_JSON
dest.parent.mkdir(parents=True, exist_ok=True)
all_ = json.loads(P.thresholds_json().read_text()) if P.thresholds_json().exists() else {}
all_.update(out)
json.dump(all_, open(dest, "w"), indent=1)
print("wrote", dest)
