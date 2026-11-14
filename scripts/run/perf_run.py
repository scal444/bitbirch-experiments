"""One timed run (time and peak memory against N); appends one row to data/timing/timings.csv of the results folder.

  python scripts/run/perf_run.py nvmolkit   SET [--repeats 2]              nvMolKit batched (GPU, device-resident input)
  python scripts/run/perf_run.py serial     SET [--repeats 1]              bblean serial, one thread
  python scripts/run/perf_run.py batched    SET --threads T [--repeats 2]  bblean batched (multithreaded CPU), T threads
  python scripts/run/perf_run.py multiround SET --ps P [--repeats 1]       bblean split-merge (bb multiround), P processes

Builds: serial and multiround from BB_BBLEAN_TREE, batched from BB_BBLEAN_BATCHED, nvmolkit from BB_NVMOLKIT_PY.
All at k = 3.5, bf 1024 (or --bf), batch 4096, input in the seed-0 order (run scripts/run/perf_inputs.py first).
Time: wall time of the clustering call only (fit / bitbirch), best of --repeats; multiround: wall time of the whole
`bb multiround` command on the pre-split input dir (process start, file reads and all three rounds).
Memory:
  host_peak_gib  serial / batched / nvmolkit: peak RSS of this process (getrusage), which includes the input copy
                 (input_gib); multiround: bblean's own monitor, sum of RSS over all its processes, sampled every 1 s.
  gpu_pool_peak_gib  nvmolkit: CUDA default memory pool UsedMemHigh during the call (nvMolKit allocates there with
                 cudaMallocAsync); the device input (gpu_input_gib, torch allocator) is extra.
Run it on an otherwise idle machine.
"""
import argparse
import csv
import ctypes
import datetime
import json
import os
import pickle
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import paths as P  # noqa: E402
from scripts.run import perf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("player", choices=["nvmolkit", "serial", "batched", "multiround"])
ap.add_argument("set", choices=list(perf.SETS_N))
ap.add_argument("--threads", type=int, default=36)
ap.add_argument("--ps", type=int, default=36)
ap.add_argument("--bf", type=int, default=1024)
ap.add_argument("--k", type=float, default=3.5)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--batch-size", type=int, default=4096)
ap.add_argument("--out", default=str(P.TIMING / "timings.csv"))
ap.add_argument("--tag", default="")
a = ap.parse_args()

os.environ.setdefault("BITBIRCH_BUILD_CPP", "1")
s = a.set
n = perf.SETS_N[s]
base = perf.base_of(s)
_t = P.thresholds()[base]
t = _t["isim"] + a.k * _t["sigma_n50"]
row = dict(player=a.player, set=s, n=n, k=a.k, threshold=round(t, 6), bf=a.bf, batch_size="", threads="", processes="",
           repeats=a.repeats, seconds_min="", seconds_all="", clusters="", host_peak_gib="", input_gib=round(n * 256 / 2**30, 3),
           gpu_pool_peak_gib="", gpu_input_gib="", build="", commit="", tag=a.tag)
secs = []

if a.player in ("serial", "batched"):
    root = P.path("BB_BBLEAN_TREE" if a.player == "serial" else "BB_BBLEAN_BATCHED")
    sys.path.insert(0, str(root))
    import bblean
    assert Path(bblean.__file__).parent.parent == root, bblean.__file__
    x = perf.load(s)
    kw = dict(batch_size=a.batch_size, n_threads=a.threads) if a.player == "batched" else {}
    for _ in range(a.repeats):
        tree = None
        tree = bblean.BitBirch(threshold=t, branching_factor=a.bf, merge_criterion="diameter", **kw)
        st = time.perf_counter()
        tree.fit(x)
        secs.append(time.perf_counter() - st)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # before building the python cluster lists
    row.update(build=str(root), commit=perf.git_commit(root), host_peak_gib=round(peak, 3),
               clusters=len(tree.get_cluster_mol_ids()))
    if a.player == "batched":
        row.update(batch_size=a.batch_size, threads=a.threads)
    else:
        row.update(threads=1)

elif a.player == "multiround":
    root = P.path("BB_BBLEAN_TREE")
    d = perf.parts_dir(s, a.ps)
    assert d.exists(), f"run perf_inputs.py {s} --parts {a.ps} first"
    env = dict(os.environ, PYTHONPATH=str(root))
    out = P.OUT / "tmp" / f"perf_mr_{s}_{a.ps}"
    for _ in range(a.repeats):
        shutil.rmtree(out, ignore_errors=True)
        cmd = [sys.executable, "-c", "import sys; from bblean.cli import app; sys.argv[0]='bb'; app()",
               "multiround", str(d), "-o", str(out), "--ps", str(a.ps), "-b", str(a.bf), "-t", repr(t)]
        st = time.perf_counter()
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        secs.append(time.perf_counter() - st)
        rounds = json.loads((out / "timings.json").read_text())
        rss = json.loads((out / "max-rss.json").read_text())
    row.update(processes=a.ps, build=str(root), commit=perf.git_commit(root),
               clusters=len(pickle.loads((out / "clusters.pkl").read_bytes())),
               host_peak_gib=round(float(rss if not isinstance(rss, dict) else max(rss.values())), 3),
               tag=(a.tag + " " if a.tag else "") + "rounds=" + perf.dumps({k: round(v, 2) for k, v in rounds.items()}))
    shutil.rmtree(out, ignore_errors=True)

else:  # nvmolkit
    import numpy as np
    root = P.path("BB_NVMOLKIT_PY")
    sys.path.insert(0, str(root))
    import torch
    from nvmolkit.clustering import bitbirch
    x_host = perf.load(s).view(np.int32)
    torch.zeros(1, device="cuda")
    cu = ctypes.CDLL("libcuda.so.1")
    pool = ctypes.c_void_p()
    assert cu.cuDeviceGetDefaultMemPool(ctypes.byref(pool), 0) == 0

    def pool_attr(i):
        v = ctypes.c_uint64(); assert cu.cuMemPoolGetAttribute(pool, i, ctypes.byref(v)) == 0; return v.value

    peaks = []
    bitbirch(torch.from_numpy(x_host[:100_000]).cuda(), t, branching_factor=a.bf, batch_size=a.batch_size).numpy()
    xg = torch.from_numpy(x_host).cuda()
    for _ in range(a.repeats):
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        cu.cuMemPoolTrimTo(pool, ctypes.c_size_t(0))
        z = ctypes.c_uint64(0); cu.cuMemPoolSetAttribute(pool, 8, ctypes.byref(z))  # reset UsedMemHigh
        st = time.perf_counter()
        lab = bitbirch(xg, t, branching_factor=a.bf, batch_size=a.batch_size).numpy()
        secs.append(time.perf_counter() - st)
        peaks.append(pool_attr(8))
    del xg
    torch.cuda.empty_cache()
    row.update(batch_size=a.batch_size, build=str(root), commit=perf.git_commit(P.path("BB_NVMOLKIT_PY").parent /
                                                                                "nvmolkit-main"),
               clusters=int(lab.max() + 1), gpu_pool_peak_gib=round(max(peaks) / 2**30, 3),
               gpu_input_gib=row["input_gib"], gpu=torch.cuda.get_device_name(),
               host_peak_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))

row.update(seconds_min=round(min(secs), 4), seconds_all=" ".join(f"{v:.4f}" for v in secs),
           when=datetime.datetime.now().isoformat(timespec="seconds"))
row.pop("gpu", None)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
new = not out.exists()
cols = list(row) if new else next(csv.reader(open(out)))  # older files have extra columns (load_before, superseded)
with open(out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, restval="")
    if new:
        w.writeheader()
    w.writerow(row)
print(perf.dumps(row), flush=True)
