"""MoleculeACE metrics: CH / DB / Dunn (paper_indices.py, Tanimoto/iSIM versions of the BitBIRCH paper) and cluster
counts for every labelling made by scripts/run/run_moleculeace.py (BB_OUT/moleculeace/labels).

   python scripts/metrics/moleculeace_metrics.py      (GPU; incremental by set x method)

Output: data/moleculeace/moleculeace_indices.csv of the results folder, one row per set x method x threshold x min_size
(columns in the results folder README).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.common import paths as P  # noqa: E402
import paper_indices as PI  # noqa: E402

OUT = P.MOLECULEACE / "moleculeace_indices.csv"
root = P.OUT / "moleculeace" / "labels"
old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["set", "method"])
done = set(zip(old["set"], old["method"]))
rows = []
for npz in sorted(root.glob("*/*.npz")):
    s, method = npz.parent.name, npz.stem
    if (s, method) in done or npz.name.endswith(".partial.npz"):
        continue
    fps = np.load(P.DATA / "moleculeace" / s / "rdkit_2048.npy")
    lab = np.load(npz)
    for key in sorted(lab.files):
        labels = np.unique(lab[key], return_inverse=True)[1].astype(np.int64)
        sizes = np.bincount(labels)
        res = PI.indices(fps, labels)
        for r in res:
            rows.append(dict(set=s, method=method, n=len(fps), threshold=float(key[1:]), clusters=len(sizes),
                             singletons=int((sizes == 1).sum()), **r))
    print(s, method, flush=True)
df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.sort_values(["set", "method", "threshold", "min_size"]).to_csv(OUT, index=False)
print(len(df), "rows ->", OUT)
