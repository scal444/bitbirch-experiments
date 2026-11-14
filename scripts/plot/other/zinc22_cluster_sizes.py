"""In the form of the BitBIRCH paper (2025) Fig 7A: density of the 10,000 largest cluster sizes, the 12.16B ZINC22 run.

nvMolKit batched, ECFP4 2048, k = 3.5, bf 1024
(the paper: 1B ZINC22, RDKit FP, BitBIRCH parallel). Gaussian KDE like the original.
Data: data/zinc22/largest_cluster_sizes.csv (the 10,000 largest cluster sizes of the ZINC22 run).
Output: zinc22_cluster_sizes.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
import common as C  # noqa: E402

TOP = 10_000


def main():
    C.style()
    f = C.P.ZINC22 / "largest_cluster_sizes.csv"
    if not f.exists():
        C.waiting("zinc22_cluster_sizes", f"{f} missing"); return
    sizes = pd.read_csv(f).sort_values("rank")["size"].to_numpy()[:TOP].astype(float)
    x = np.linspace(0, sizes.max() * 1.02, 600)
    dens = gaussian_kde(sizes)(x)
    fig, ax = plt.subplots(figsize=(C.COLW, 1.9))
    ax.plot(x / 1e6, dens * 1e6, color=C.NV)
    ax.fill_between(x / 1e6, dens * 1e6, color=C.NV, alpha=0.25, lw=0)
    ax.set_xlabel("cluster size (millions of molecules)")
    ax.set_ylabel("density (per million)")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.set_title("ZINC22, 10,000 largest clusters")
    C.save(fig, "zinc22_cluster_sizes")


if __name__ == "__main__":
    main()
