"""Small adapter used by metrics.py and validate_metrics.py: paths, datasets and
thresholds from scripts/common/paths.py under the names those scripts use.

BP -> BB_OUT: labels are read from BB_OUT/labels/<ds>/<method>/k<k>/s<seed>.npy, metrics go to data/quality/by_run/ of the results folder.
label_path() falls back to earlier labels in BB_BP_LABELS (optional) when a file is not under BB_OUT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as _P  # noqa: E402

HOME = Path.home()
BP = _P.OUT
DATA = _P.DATA
BF = _P.BF
KS = _P.KS
DATASETS = _P.DATASETS
ACTIVES = _P.path("BB_ACTIVES")
threshold, sigma, kstr, perm = _P.threshold, _P.sigma, _P.kstr, _P.perm
save = _P.save_labels


def isim(ds):
    return _P.isim(ds)


def label_path(ds, method, k, seed, bf=BF):
    m = method if bf == BF else f"{method}_bf{bf}"
    p = _P.label_path(ds, m, k, seed)
    if p.exists():
        return p
    q = _P.label_path(ds, m, k, seed, root="bp")
    return q if q.exists() else p
