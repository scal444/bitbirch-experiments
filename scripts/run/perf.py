"""Shared definitions for the timing runs: sets and sizes, thresholds, input files, build commit.

Sets: ena1m, chembl34, ena10m (Enamine REAL 10.35M), zinc10m / zinc25m / zinc50m / zinc101m (the first N rows of
ZINC20 101M in the seed-0 order). Fingerprints under BB_DATA; shuffled inputs under BB_OUT/perf_inputs.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

_T = P.thresholds()
# base set: (fingerprint file, rows, t at k = 3.5)
SETS = {
    "ena1m": (P.DATASETS["ena1m"]["fps"], 1_000_000, _T["ena1m"]["t_3.5"]),
    "chembl34": (P.DATASETS["chembl34"]["fps"], 2_409_133, _T["chembl34"]["t_3.5"]),
    "ena10m": (P.DATA / "enamine_real_10m/ecfp4_2048.npy", 10_353_526, _T["ena10m"]["t_3.5"]),
    "zinc101m": (P.DATA / "zinc20ml/ecfp4_2048.npy", 101_270_110, _T["zinc101m"]["t_3.5"]),
}
SETS_N = {"ena1m": 1_000_000, "chembl34": 2_409_133, "ena10m": 10_353_526, "zinc10m": 10_000_000,
          "zinc25m": 25_000_000, "zinc50m": 50_000_000, "zinc101m": 101_270_110}
INPUTS = P.OUT / "perf_inputs"


def base_of(s):
    return "zinc101m" if s.startswith("zinc") else s


def threshold(s):
    return SETS[base_of(s)][2]


def shuffled_path(s):
    return INPUTS / f"{base_of(s)}_s0.npy"


def parts_dir(s, ps):
    return INPUTS / f"{s}_parts{ps}"


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def load(s):
    """The set in RAM (a real copy), seed-0 order."""
    import numpy as np
    n = SETS_N[s]
    with open(shuffled_path(s), "rb") as fh:  # plain read (no memory map, so RSS counts the input once)
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
        assert not fortran and dtype == np.uint8 and shape[0] >= n, (shape, dtype)
        x = np.fromfile(fh, dtype=np.uint8, count=n * shape[1])
    return x.reshape(n, shape[1])


def dumps(x):
    return json.dumps(x, separators=(",", ":"))
