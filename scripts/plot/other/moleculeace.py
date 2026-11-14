"""Shared loader for the MoleculeACE runs: 30 ChEMBL sets x thresholds 0.30-0.80 x implementations, bf 50.

Data: data/moleculeace/moleculeace_indices.csv (scripts/metrics/moleculeace_metrics.py), one row per
set x method x threshold x min_size with `clusters_kept` (clusters with at least min_size members), `chi`
(Calinski-Harabasz), `dbi` (Davies-Bouldin) and `dunn`, all on clusters with at least min_size members.
Methods: bblean_tree (bblean serial), bblean_batched (bblean batched, multithreaded CPU), bblean_mr10 (bblean split-merge,
10 files / 10 processes as in the BitBIRCH paper), nvmolkit_main (nvMolKit batched). Rows of other
methods in the CSV (an nvMolKit Butina reference) are not drawn. Not a figure itself (make_all.py skips it).
Davies-Bouldin is set to NaN where the row has zero-distance centroid pairs (undefined there).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import common as C  # noqa: E402

F = C.P.MOLECULEACE / "moleculeace_indices.csv"
PLAYER = {"bblean_tree": "serial", "bblean_mr10": "multiround", "nvmolkit_main": "batched",
          "bblean_batched": "bblean_batched"}
QSTYLE = {"serial": C.QP["serial"], "multiround": dict(C.QP["multiround"], label=f"{C.PAR_NAME} (P=10)"),
          "bblean_batched": C.QP["bblean_batched"], "batched": C.QP["batched"]}



def mkw(p, ms=3):
    """Marker kwargs: bblean batched as a ring under the smaller nvMolKit dot (identical clusters), else plain markers."""
    return C.overlay(p, ms / 4) if p in C.OVERLAY else dict(ms=ms)


def load():
    if not F.exists():
        return pd.DataFrame()
    d = pd.read_csv(F)
    d["player"] = d["method"].map(PLAYER)
    d = d[d.player.notna()].rename(columns={"chi": "ch", "dbi": "db"})
    # Davies-Bouldin is undefined when two kept clusters have the same centroid (distance 0); those rows
    # (about 3 %) carry huge values, so DB is left out for them
    if "dbi_zero_distance_pairs" in d:
        d.loc[d.dbi_zero_distance_pairs > 0, "db"] = float("nan")
    return d
