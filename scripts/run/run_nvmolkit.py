"""Quality labels from nvMolKit batched (GPU). Needs a GPU and the nvMolKit build at BB_NVMOLKIT_PY (see setup/).

  python scripts/run/run_nvmolkit.py DS [K ...] [--seeds 0 1] [--batch-size 4096] [--bf 1024]
         [--repeats 1] [--method nvmolkit_main] [--compare-bp]

Writes BB_OUT/labels/DS/METHOD/kK/sSEED.npy (+ .json with threshold, seconds, GPU memory). Times only the
clustering call on a GPU-resident input. --compare-bp (optional) records whether the partition equals earlier labels in
BB_BP_LABELS/DS/nvmolkit, when those exist.
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402

sys.path.insert(0, str(P.path("BB_NVMOLKIT_PY")))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from nvmolkit.clustering import bitbirch  # noqa: E402

from scripts.common.partitions import same_partition  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("ks", nargs="*", type=float, default=list(P.KS))
ap.add_argument("--seeds", nargs="*", type=int)
ap.add_argument("--batch-size", type=int, default=4096)
ap.add_argument("--bf", type=int, default=P.BF)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--method", default="nvmolkit_main")
ap.add_argument("--compare-bp", action="store_true")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

torch.zeros(1, device="cuda")
_cu = ctypes.CDLL("libcuda.so.1")
_pool = ctypes.c_void_p()
assert _cu.cuDeviceGetDefaultMemPool(ctypes.byref(_pool), 0) == 0


def pool_reset():
    """Reset the CUDA default pool's UsedMemHigh (nvMolKit allocates there with cudaMallocAsync)."""
    _cu.cuMemPoolTrimTo(_pool, ctypes.c_size_t(0))
    z = ctypes.c_uint64(0)
    _cu.cuMemPoolSetAttribute(_pool, 8, ctypes.byref(z))


def pool_peak():
    v = ctypes.c_uint64()
    _cu.cuMemPoolGetAttribute(_pool, 8, ctypes.byref(v))
    return v.value


d = P.DATASETS[a.ds]
words = np.ascontiguousarray(np.load(d["fps"])).view(np.int32)
n = len(words)
for k in a.ks:
    for s in a.seeds if a.seeds is not None else d["seeds"]:
        out = P.label_path(a.ds, a.method, k, s)
        if out.exists() and not a.force:
            continue
        perm = P.perm(n, s)
        t = P.threshold(a.ds, k)
        x_host = np.ascontiguousarray(words[perm])
        secs, peaks = [], []
        x = torch.from_numpy(x_host).cuda()
        for _ in range(a.repeats):
            torch.cuda.synchronize()
            pool_reset()
            st = time.perf_counter()
            lab = bitbirch(x, t, branching_factor=a.bf, batch_size=a.batch_size).numpy()
            secs.append(time.perf_counter() - st)
            peaks.append(pool_peak())
        del x
        torch.cuda.empty_cache()
        full = np.empty(n, dtype=np.int32)
        full[perm] = lab
        info = dict(threshold=t, bf=a.bf, batch_size=a.batch_size, seconds=min(secs), seconds_all=secs,
                    clusters=int(full.max() + 1), gpu=torch.cuda.get_device_name(),
                    gpu_pool_peak_gib=max(peaks) / 2**30)
        if a.compare_bp:
            ref = P.label_path(a.ds, "nvmolkit", k, s, root="bp")
            if ref.exists():
                info["same_as_bp_nvmolkit"] = bool(same_partition(full, np.load(ref)))
        P.save_labels(out, full, info)
        print(a.ds, a.method, f"k{k:g}", f"s{s}", info, flush=True)
