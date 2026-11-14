"""Paths, datasets and thresholds for every script in this repo.

Reads config/paths.env and stops if it's missing (copy config/paths.example.env first); any key can be
overridden by an environment variable of the same name. Exception: when BB_RESULTS_NAME is set in the environment
(make_all.py does this to redraw the example results), a missing paths.env falls back to paths.example.env. Per-dataset fingerprint and
scaffold files can be overridden with BB_FPS_<DS> / BB_SCAF_<DS> (DS upper case).
Defines the results folder (results_<BB_RESULTS_NAME>), the five quality libraries (_DS below), the thresholds
t = iSIM + k * sigma, the input orders and the label layout.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    cfg = {}
    f = REPO / "config" / "paths.env"
    if not f.exists():
        if "BB_RESULTS_NAME" not in os.environ:
            sys.exit("config/paths.env not found: run `cp config/paths.example.env config/paths.env` and edit it")
        f = REPO / "config" / "paths.example.env"
    for line in f.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg.update({k: v for k, v in os.environ.items() if k.startswith("BB_")})
    return cfg


CFG = _load()


def path(key):
    return Path(os.path.expanduser(CFG[key]))


DATA, OUT = path("BB_DATA"), path("BB_OUT")
# Results folder: <repo>/results_<BB_RESULTS_NAME>/, with data/ (CSV and JSON written by the metric, timing and
# refinement scripts) and plots/ (figures written by scripts/plot/make_all.py). Set BB_RESULTS_NAME in
# config/paths.env (or the environment) to a short description of your machine or run to get your own folder.
DEFAULT_RESULTS_NAME = "gb300_72coreARM"  # the results that come with the repository (written only if named)
RESULTS_NAME = CFG.get("BB_RESULTS_NAME", "my_machine")
RESULTS = REPO / f"results_{RESULTS_NAME}"
RES_DATA = RESULTS / "data"
PLOTS = RESULTS / "plots"
QUALITY = RES_DATA / "quality"            # metrics per run (by_run/), combined tables, largest clusters
TIMING = RES_DATA / "timing"              # timings.csv, scaling_exponents.csv
REFINEMENT = RES_DATA / "refinement"      # refinement experiment tables
MOLECULEACE = RES_DATA / "moleculeace"    # MoleculeACE runs and indices
ZINC22 = RES_DATA / "zinc22"              # the 12.16 billion molecule ZINC22 run
THRESHOLDS = RES_DATA / "thresholds"      # isim_sigma.json: iSIM and sigma per library
CHECKS = RES_DATA / "checks"              # runs that must give identical clusters do


BF = 1024
KS = (3.0, 3.5, 4.0)

# The libraries of the quality comparison. name: (fingerprint file, scaffold-id file, input orders), files relative
# to BB_DATA. Fingerprints: packed uint8 .npy, 2048 bits = 256 bytes per row; scaffold ids: one integer per row
# (docs/DATA.md).
_DS = {
    "chembl34": ("chembl34/ecfp4_2048.npy", "chembl34/murcko_scaffold_ids.npy", range(3)),
    "scu": ("scubidoo/ecfp4_2048.npy", "scubidoo/murcko_scaffold_ids.npy", range(3)),
    "ena1m": ("enamine_real_10m/subset_1m/ecfp4_2048.npy", "enamine_real_10m/subset_1m/murcko_scaffold_ids.npy", range(3)),
    "c35": ("chembl35_new/ecfp4_2048.npy", "chembl35_new/murcko_scaffold_ids.npy", range(5)),
    "np": ("chembl33_np/rdkit_2048.npy", "chembl33_np/murcko_scaffold_ids.npy", range(5)),
}
DATASETS = {
    ds: dict(
        fps=Path(os.path.expanduser(CFG.get(f"BB_FPS_{ds.upper()}", DATA / f))),
        scaf=Path(os.path.expanduser(CFG.get(f"BB_SCAF_{ds.upper()}", DATA / s))),
        seeds=seeds,
    )
    for ds, (f, s, seeds) in _DS.items()
}

# Thresholds t_k = iSIM + k * iSIM-sigma (bblean estimate_jt_std, n_samples=50), from data/thresholds/isim_sigma.json
# of the results folder (scripts/metrics/thresholds.py writes it). A new results folder without that file uses the
# values of the results that come with the repository, so the five libraries need no recomputation.
THRESHOLDS_JSON = THRESHOLDS / "isim_sigma.json"
_T_CACHE = {}


def thresholds_json():
    """isim_sigma.json in use: this results folder's, else the one of the results that come with the repository."""
    if THRESHOLDS_JSON.exists():
        return THRESHOLDS_JSON
    return REPO / f"results_{DEFAULT_RESULTS_NAME}" / "data" / "thresholds" / "isim_sigma.json"


def thresholds():
    if not _T_CACHE:
        _T_CACHE.update(json.loads(thresholds_json().read_text()))
    return _T_CACHE


def isim(ds):
    return thresholds()[ds]["isim"]


def threshold(ds, k):
    return thresholds()[ds]["isim"] + k * thresholds()[ds]["sigma_n50"]


def sigma(ds):
    return thresholds()[ds]["sigma_n50"]


def kstr(k):
    return f"k{k:g}"


def label_path(ds, method, k, seed):
    """Labels under BB_OUT/labels as int32 .npy (cluster id per original row) + .json with run info."""
    return OUT / "labels" / ds / method / kstr(k) / f"s{seed}.npy"


def perm(n, seed):
    """Input order used everywhere: numpy default_rng(seed).permutation(n)."""
    import numpy as np
    return np.random.default_rng(seed).permutation(n)


def save_labels(p, labels, info):
    import numpy as np
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".partial.npy")
    np.save(tmp, labels.astype(np.int32))
    tmp.rename(p)
    p.with_suffix(".json").write_text(json.dumps(info, indent=1))
