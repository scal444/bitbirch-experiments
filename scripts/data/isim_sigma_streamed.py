"""Clustering thresholds iSIM + k * iSIM-sigma (the best-practices paper's recipe, bblean functions) for a fingerprint
file larger than RAM. Used for the ZINC20-ML 101M set (data/thresholds/isim_sigma.json of the results folder, key zinc101m).

usage: python scripts/data/isim_sigma_streamed.py FPS.npy OUT.json   (see scripts/metrics/thresholds.py for sets that fit in memory)
iSIM over ALL rows from per-bit column sums (bblean.similarity.jt_isim_from_sum; identical to jt_isim_packed).
Note: jt_isim_from_sum overflows uint64 once column sums pass ~4.3e9 (billions of rows); for ZINC22 use
zinc22_buckets.py isim, which uses exact integers.
sigma = bblean estimate_jt_std(fps, n_samples=50); bblean itself first draws a 1M random subset (rng seed 42) when N > 1M.
"""
import json, sys, time
import numpy as np
import bblean.similarity as S

src, out = sys.argv[1], sys.argv[2]
fps = np.load(src, mmap_mode="r")
n = len(fps)
t0 = time.time()
sums = np.zeros(fps.shape[1] * 8, dtype=np.uint64)
step = 2_000_000
for b in range(0, n, step):
    sums += np.unpackbits(np.asarray(fps[b:b + step]), axis=1, bitorder="big").sum(axis=0, dtype=np.uint64)
    print(f"colsum {min(b + step, n):,}/{n:,} {time.time() - t0:.0f}s", flush=True)
isim = float(S.jt_isim_from_sum(sums, n))
t1 = time.time()
sigma = float(S.estimate_jt_std(fps, n_samples=50))
t2 = time.time()
r = dict(src=src, n=n, isim=isim, sigma_n50=sigma, **{f"t_{k}": isim + k * sigma for k in (3, 3.5, 4)},
         isim_secs=t1 - t0, sigma_secs=t2 - t1)
print(r, flush=True)
json.dump(r, open(out, "w"), indent=1)
