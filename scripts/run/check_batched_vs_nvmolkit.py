"""Check that bblean batched (multithreaded CPU) and nvMolKit batched (GPU) labels are the same partition for given input orders.

  python scripts/run/check_batched_vs_nvmolkit.py DS K SEED [SEED ...]

Reads BB_OUT/labels/DS/{bblean_batched,nvmolkit_main}/kK/sSEED.npy. Appends one row per order to
data/checks/bblean_batched_vs_nvmolkit.csv (results folder) and exits 1 if any order differs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scripts.common import paths as P  # noqa: E402
from scripts.common.partitions import same_partition  # noqa: E402

ds, k, seeds = sys.argv[1], float(sys.argv[2]), [int(s) for s in sys.argv[3:]]
rows = []
for s in seeds:
    a, b = np.load(P.label_path(ds, "bblean_batched", k, s)), np.load(P.label_path(ds, "nvmolkit_main", k, s))
    rows.append(dict(dataset=ds, k=k, seed=s, clusters_bblean_batched=int(a.max() + 1), clusters_nvmolkit=int(b.max() + 1),
                     same_partition=bool(same_partition(a, b))))
    print(rows[-1], flush=True)
out = P.CHECKS / "bblean_batched_vs_nvmolkit.csv"
out.parent.mkdir(parents=True, exist_ok=True)
d = pd.DataFrame(rows)
d.to_csv(out, mode="a", header=not out.exists(), index=False)
sys.exit(0 if d.same_partition.all() else 1)
