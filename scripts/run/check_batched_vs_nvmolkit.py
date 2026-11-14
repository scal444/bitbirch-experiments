"""Check that bblean batched (multithreaded CPU) and nvMolKit batched (GPU) labels are the same partition.

  python scripts/run/check_batched_vs_nvmolkit.py [DS K ORDER [ORDER ...]]

Without arguments: every library, k and input order that has labels from both under BB_OUT/labels. With arguments:
those orders only (a pair with a missing label file is reported and skipped). Writes one row per pair to
data/checks/bblean_batched_vs_nvmolkit.csv of the results folder (a row for the same library, k and order is
replaced) and exits 1 if any pair differs. `nvmolkit_build` is the nvMolKit commit stored with the labels, if any.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import label_folder  # noqa: E402
from scripts.common.partitions import same_partition  # noqa: E402

KEY = ["dataset", "k", "seed"]
BB, NV = label_folder("bblean_batched"), label_folder("nvmolkit_batched")
if len(sys.argv) > 1:
    jobs = [(sys.argv[1], float(sys.argv[2]), int(s)) for s in sys.argv[3:]]
else:
    jobs = [(ds, k, s) for ds in P.DATASETS for k in P.KS for s in range(20) if P.label_path(ds, NV, k, s).exists()]
rows = []
for ds, k, s in jobs:
    fa, fb = P.label_path(ds, BB, k, s), P.label_path(ds, NV, k, s)
    if not (fa.exists() and fb.exists()):
        print(f"{ds} k{k:g} s{s}: missing {fa if not fa.exists() else fb}, skipped", flush=True)
        continue
    a, b = np.load(fa), np.load(fb)
    info = json.loads(fb.with_suffix(".json").read_text()) if fb.with_suffix(".json").exists() else {}
    rows.append(dict(dataset=ds, k=k, seed=s, clusters_bblean_batched=len(np.unique(a)),
                     clusters_nvmolkit_batched=len(np.unique(b)), nvmolkit_build=info.get("commit", ""),
                     same_partition=bool(same_partition(a, b))))
    print(rows[-1], flush=True)
if not rows:
    sys.exit("no pairs of bblean batched and nvMolKit batched labels found")
out = P.CHECKS / "bblean_batched_vs_nvmolkit.csv"
out.parent.mkdir(parents=True, exist_ok=True)
new = pd.DataFrame(rows)
old = pd.read_csv(out, dtype={"nvmolkit_build": str}, keep_default_na=False) if out.exists() else new.iloc[:0]
old = old[~old[KEY].apply(tuple, axis=1).isin(set(new[KEY].apply(tuple, axis=1)))]
pd.concat([old, new], ignore_index=True).sort_values(KEY).to_csv(out, index=False)
sys.exit(0 if new.same_partition.all() else 1)
