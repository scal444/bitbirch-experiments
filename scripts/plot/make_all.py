"""Regenerate every figure from the tables under data/ of the results folder; print what was written and what waits.

  python scripts/plot/make_all.py [--metrics]

The results folder is results_<BB_RESULTS_NAME> (config/paths.env). First rebuilds data/quality/quality.csv and
top_clusters_by_run.csv from the per-run metrics (scripts/metrics/combine_quality.py, CSV only), so the combined
tables never lag behind. --metrics also refreshes the largest-cluster tables (scripts/metrics/top_clusters.py,
needs the labels under BB_OUT). Then runs every script in scripts/plot/quality, timing and other; each writes its
figures to plots/<topic>/<level>/ as .png (placement: FIGURES in scripts/plot/common.py). Finally writes
plots/INDEX.md, one line per figure. A figure whose data is missing is reported as waiting, not drawn.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SKIP = {"moleculeace.py"}  # shared loader, not a figure
TOPICS = ("quality", "timing", "other")
LEVELS = ("primary", "secondary", "extra")


def write_index():
    """plots/INDEX.md: one line per figure file present (name, what it shows, data it uses)."""
    lines = [f"# Figures in {C.RESULTS.name}", "",
             "Written by `python scripts/plot/make_all.py`. Each figure is a .png in "
             "`plots/<topic>/<level>/`; data paths are relative to this results folder.", ""]
    for topic in TOPICS:
        for level in LEVELS:
            folder = C.PLOTS / topic / level
            names = sorted(f.stem for f in folder.glob("*.png")) if folder.is_dir() else []
            lines += [f"## {topic} / {level}", ""]
            if not names:
                lines += ["(none)", ""]
                continue
            lines += ["| figure | what it shows | data |", "|---|---|---|"]
            for n in names:
                _, (_, _, what, data) = C.placement(n)
                lines.append(f"| `{topic}/{level}/{n}` | {what} | {data.replace('<ds>', '<library>')} |")
            lines.append("")
    C.PLOTS.mkdir(parents=True, exist_ok=True)
    (C.PLOTS / "INDEX.md").write_text("\n".join(lines))
    return C.PLOTS / "INDEX.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", action="store_true")
    a = ap.parse_args()
    print("results folder:", C.RESULTS, flush=True)
    subprocess.run([sys.executable, str(REPO / "scripts/metrics/combine_quality.py")], check=True)
    if a.metrics:
        subprocess.run([sys.executable, str(REPO / "scripts/metrics/top_clusters.py")], check=False)
    for topic in TOPICS:
        for level in LEVELS:
            (C.PLOTS / topic / level).mkdir(parents=True, exist_ok=True)
    done, waiting = [], []
    for f in [f for t in TOPICS for f in sorted((HERE / t).glob("*.py"))]:
        if f.name in SKIP:
            continue
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
        out = r.stdout + r.stderr
        rel = f.relative_to(HERE)
        if r.returncode:
            waiting.append(f"{rel}: FAILED\n{out[-800:]}")
        for line in out.splitlines():
            if line.startswith("WAITING"):
                waiting.append(f"{rel}: {line[8:]}")
            elif line.startswith("wrote"):
                done.append(line[6:])
    print(f"\n{len(done)} figures written")
    print("\n".join(done))
    print(f"\nwrote {write_index()}")
    print(f"\n{len(waiting)} waiting / failed")
    print("\n".join(waiting))


if __name__ == "__main__":
    main()
