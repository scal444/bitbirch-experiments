"""MoleculeACE metrics: CH / DB / Dunn (paper_indices.py, Tanimoto/iSIM versions of the BitBIRCH paper) and cluster
counts for every labelling made by scripts/run/run_moleculeace.py (BB_OUT/moleculeace/labels).

   python scripts/metrics/moleculeace_metrics.py      (GPU; incremental by set x implementation)

Output: data/moleculeace/moleculeace_indices.csv of the results folder, one row per set x implementation x threshold x
min_size (columns in the results folder README). Label files of other names are skipped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.common import paths as P  # noqa: E402
from scripts.common.implementations import from_folder  # noqa: E402
import paper_indices as PI  # noqa: E402

OUT = P.MOLECULEACE / "moleculeace_indices.csv"
root = P.OUT / "moleculeace" / "labels"
old = pd.read_csv(OUT, dtype={"processes": "Int64"}) if OUT.exists() else \
    pd.DataFrame(columns=["set", "implementation", "processes"])
done = {(s, i, None if pd.isna(p) else int(p)) for s, i, p in zip(old["set"], old["implementation"], old["processes"])}
rows = []
for npz in sorted(root.glob("*/*.npz")):
    s, impl = npz.parent.name, from_folder(npz.stem)
    if impl is None or (s, *impl) in done or npz.name.endswith(".partial.npz"):
        continue
    fps = np.load(P.DATA / "moleculeace" / s / "rdkit_2048.npy")
    lab = np.load(npz)
    for key in sorted(lab.files):
        labels = np.unique(lab[key], return_inverse=True)[1].astype(np.int64)
        sizes = np.bincount(labels)
        res = PI.indices(fps, labels)
        for r in res:
            rows.append(dict(set=s, implementation=impl[0], processes=impl[1], n=len(fps), threshold=float(key[1:]),
                             clusters=len(sizes), singletons=int((sizes == 1).sum()), **r))
    print(s, *impl, flush=True)
new = pd.DataFrame(rows, columns=list(old.columns) if len(old) else None)
if len(new):
    new = new.astype({"processes": "Int64"})
df = pd.concat([old, new], ignore_index=True) if len(old) else new
OUT.parent.mkdir(parents=True, exist_ok=True)
df.sort_values(["set", "implementation", "processes", "threshold", "min_size"]).to_csv(OUT, index=False)
print(len(df), "rows ->", OUT)
