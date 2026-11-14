"""Headline table: per dataset and procedure, agreement before (it0) and after the last iteration -> data/refinement/summary.csv

   python scripts/refinement/summarize.py [--paths cfg.json]
Columns (seed means where several seeds exist):
  serial_batched_*      bblean serial vs batched (same seed)
  serial_parallel_*, parallel_batched_*   with bblean split-merge (bb multiround, P = 36)
  serial_seeds_ami      bblean serial vs itself with another input order (seed pairs)
  batched_seeds_ami     batched vs itself with another input order
  <player>_vs_input_ami refined result vs the same implementation's unrefined result
  <player>_fixed_point  last iteration identical to the one before (seed 0)
Also data/refinement/quality_summary.csv: per dataset, procedure, iteration (0 = unrefined, and the last) and implementation,
the quality metrics averaged over the seeds that have them (n_seeds): clusters, singletons, largest cluster,
size-weighted within-cluster iSIM, scaffold purity, medoid iSIM, scaffold AMI, target AMI (chembl34, c35).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--paths"); a = ap.parse_args()
    cfg = C.load_cfg(a.paths); out = Path(cfg["csv_dir"])
    rows = []
    for f in sorted(out.glob("agreement_*.csv")):
        ag = pd.read_csv(f); ds = f.stem.split("_", 1)[1]
        for proc, last in C.PROCS.items():
            for it in (0, last):
                r = dict(dataset=ds, proc=proc, it=it)
                p = ag[(ag.kind == "players") & (ag.proc == proc) & (ag.it == it)]
                for pa, pb in (("serial", "batched"), ("serial", "parallel"), ("parallel", "batched")):
                    s = p[(p.player_a == pa) & (p.player_b == pb)]
                    r[f"{pa}_{pb}_ami"] = s.ami.mean(); r[f"{pa}_{pb}_ari"] = s.ari.mean()
                    r[f"{pa}_{pb}_pair_jaccard"] = s.pair_jaccard.mean()
                sd = ag[(ag.kind == "seeds") & (ag.proc == proc) & (ag.it == it)]
                for pl in C.PLAYERS:
                    r[f"{pl}_seeds_ami"] = sd[sd.player_a == pl].ami.mean()
                    if it:
                        v = ag[(ag.kind == "input") & (ag.proc == proc) & (ag.it == it) & (ag.player_a == pl)]
                        r[f"{pl}_vs_input_ami"] = v.ami.mean()
                        fp = ag[(ag.kind == "previous") & (ag.proc == proc) & (ag.it == it) & (ag.player_a == pl)
                                & (ag.seed == 0)]
                        r[f"{pl}_fixed_point"] = bool(fp.identical.iloc[0]) if len(fp) else None
                r["n_seeds"] = int(ag.seed.nunique())
                rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    qcols = ["clusters", "singletons", "largest_cluster", "isim", "scaffold_purity", "medoid_isim_nonsingleton",
             "scaf_ami", "tgt_ami"]
    qs = []
    for f in sorted(out.glob("quality_*.csv")):
        if f.name == "quality_summary.csv":
            continue
        q = pd.read_csv(f)
        for proc, last in C.PROCS.items():
            sel = q[(q.proc == proc) & q.it.isin([0, last])]
            g = sel.groupby(["dataset", "proc", "it", "player"])
            m = g[[c for c in qcols if c in q.columns]].mean()
            m["n_seeds"] = g.seed.nunique()
            qs.append(m.reset_index())
    pd.concat(qs).to_csv(out / "quality_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
