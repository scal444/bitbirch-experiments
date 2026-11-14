"""Agreement with Murcko scaffolds and ChEMBL targets against the number of clusters.

Each implementation gives a different number of clusters, so each is drawn across its settings (k = 3, 3.5, 4; line
along k, mean over input orders, min-max bars) against the cluster count. Targets: ChEMBL34 and ChEMBL35 additions only.
Data: metrics CSVs. Output: ami_vs_clusters_<ds>.*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import common as C  # noqa: E402

KS = (3, 3.5, 4)  # the settings every implementation ran (ChEMBL34's wider k scan is in quality_vs_k_chembl34)
REFS = [("scaf_ami", "AMI vs scaffolds"), ("tgt_ami", "AMI vs targets")]


def main():
    C.style()
    if not C.any_metrics(topk=False):
        C.waiting("ami_vs_clusters", "per-run metrics in data/quality/by_run"); return
    for ds in C.DS_ORDER:
        d = C.load_metrics(ds, bf=C.P.BF)
        d = d[d.player.isin(list(C.QP)) & d.k.isin(KS)]
        if d.empty:
            continue
        refs = [r for r in REFS if d[r[0]].notna().any()]
        a = C.agg_seeds(d, ["clusters"] + [r[0] for r in refs])
        fig, axs = plt.subplots(1, len(refs), figsize=(C.FULLW * max(len(refs), 2) / 3, 1.9), squeeze=False)
        for ax, (col, lab) in zip(axs[0], refs):
            for p in C.players_present(a):
                s = a[a.player == p].sort_values("k")
                s = s[s[col + "_mean"].notna()]
                C.errbar(ax, s.clusters_mean / 1e3, s[col + "_mean"], s[col + "_min"], s[col + "_max"], p)
            ax.set_xlabel("clusters (thousands)")
            ax.set_ylabel(lab)
        fig.suptitle(f"Agreement with references vs cluster count, {C.DS_NAME[ds]} (k = 3, 3.5, 4)", y=1.03)
        fig.tight_layout()
        C.legend_below(fig, axs, ncol=2, y=0.02)
        C.save(fig, f"ami_vs_clusters_{ds}")


if __name__ == "__main__":
    main()
