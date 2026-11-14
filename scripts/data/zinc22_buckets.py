"""ZINC22 2D: random-order buckets from zinc22_fingerprint.py shards, the threshold, and the bucket reader used for
clustering.

  zinc22_buckets.py route SHARD_DIR BUCKET_DIR [--buckets 64] [--seed 0] [--delete-shards]
      Shards in file order; every row gets bucket = rng([seed, file_index]).integers(buckets). Rows are appended to
      bucket files b.cnt / b.idx / b.origin (int64 = file_index << 32 | line). After each shard, the bucket file sizes
      are committed to BUCKET_DIR/commit.json; a rerun truncates every bucket file to the last commit and continues, so
      a crash never leaves a half-routed shard. --delete-shards removes a shard's .cnt/.idx/.line after its commit
      (keeps disk peak at ~1x sparse size instead of 2x).
      The report used --buckets 128 --seed 0.
  zinc22_buckets.py isim SHARD_DIR OUT.json [--k 3.5]
      iSIM over all rows from the shards' per-bit colsums (exact, no pass over the data); sigma = bblean
      estimate_jt_std(n_samples=50) on 1M rows drawn uniformly from the buckets (bblean draws its own 1M when given
      more; here the 1M are read sparse and densified first).
  zinc22_buckets.py check BUCKET_DIR
      Reads bucket 0 through read_bucket() and verifies counts and a sample of rows against the shards' bits.
read_bucket(BUCKET_DIR, b, seed) -> (packed uint32 (n, 64) rows in a seeded random order, origin int64 (n,)).
Bucket b holds ~N/buckets rows (10B / 64 = 156M rows = 40 GB packed in RAM). Feeding buckets 0..B-1, each shuffled,
is a uniformly random order of all rows.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np


def densify(cnt, idx, chunk=1_000_000):
    """Sparse rows -> packed uint32 (n, 64) in nvMolKit layout (bit i in word i // 32, position i % 32).
    ~2.5M rows/s per process (measured); chunked so the temporary (chunk, 2048) bit matrix stays ~2 GB."""
    n = len(cnt)
    out = np.empty((n, 64), np.uint32)
    ends = np.r_[0, np.cumsum(cnt.astype(np.int64))]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        c = cnt[a:b].astype(np.int64)
        rows = np.repeat(np.arange(b - a, dtype=np.int64), c)
        bits = np.zeros((b - a, 2048), np.uint8)
        bits[rows, idx[ends[a]:ends[b]]] = 1
        out[a:b] = np.packbits(bits, axis=1, bitorder="little").view(np.uint32)
    return out


def read_bucket(bdir, b, seed=0):
    bdir = Path(bdir)
    cnt = np.fromfile(bdir / f"{b:04d}.cnt", np.uint8)
    idx = np.fromfile(bdir / f"{b:04d}.idx", np.uint16)
    origin = np.fromfile(bdir / f"{b:04d}.origin", np.int64)
    packed = densify(cnt, idx)
    order = np.random.default_rng([seed, 1_000_003, b]).permutation(len(cnt))
    return packed[order], origin[order]


def route(shards, bdir, nb, seed, delete):
    bdir.mkdir(parents=True, exist_ok=True)
    commit_path = bdir / "commit.json"
    commit = json.loads(commit_path.read_text()) if commit_path.exists() else {"done": [], "sizes": {}}
    files = {}
    for b in range(nb):
        for suf in (".cnt", ".idx", ".origin"):
            p = bdir / f"{b:04d}{suf}"
            h = open(p, "ab"); h.truncate(commit["sizes"].get(p.name, 0)); h.seek(0, 2); files[p.name] = h
    done = set(commit["done"])
    t0 = time.time(); rows_total = 0
    for d in sorted(shards.glob("*.done")):
        fi = int(d.stem)
        if fi in done:
            continue
        s = shards / d.stem
        cnt = np.fromfile(f"{s}.cnt", np.uint8); idx = np.fromfile(f"{s}.idx", np.uint16)
        line = np.fromfile(f"{s}.line", np.uint32)
        bk = np.random.default_rng([seed, fi]).integers(0, nb, len(cnt))
        starts = np.r_[0, np.cumsum(cnt.astype(np.int64))]
        order = np.argsort(bk, kind="stable"); cuts = np.searchsorted(bk[order], np.arange(nb + 1))
        origin = (np.int64(fi) << 32) | line.astype(np.int64)
        for b in range(nb):
            sel = order[cuts[b]:cuts[b + 1]]
            if len(sel) == 0:
                continue
            c = cnt[sel]
            c64 = c.astype(np.int64)
            pos = np.repeat(starts[sel], c64) + (np.arange(int(c64.sum())) - np.repeat(np.r_[0, np.cumsum(c64)][:-1], c64))
            files[f"{b:04d}.cnt"].write(c.tobytes()); files[f"{b:04d}.idx"].write(idx[pos].tobytes())
            files[f"{b:04d}.origin"].write(origin[sel].tobytes())
        for h in files.values():
            h.flush(); os.fsync(h.fileno())
        done.add(fi); rows_total += len(cnt)
        commit = {"done": sorted(done), "sizes": {k: h.tell() for k, h in files.items()}, "buckets": nb, "seed": seed}
        commit_path.with_suffix(".tmp").write_text(json.dumps(commit)); os.replace(commit_path.with_suffix(".tmp"), commit_path)
        if delete:
            for suf in (".cnt", ".idx", ".line"):
                os.remove(f"{s}{suf}")
        print(f"routed shard {fi}: {len(cnt):,} rows; {rows_total:,} this run, {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


def isim(shards, out, k):
    import bblean.similarity as S
    infos = [json.loads(p.read_text()) for p in sorted(shards.glob("*.done"))]
    n = sum(i["kept"] for i in infos)
    sums = sum(np.fromfile(shards / f"{i['file_index']:05d}.colsum", np.uint64) for i in infos)
    # exact integer arithmetic: bblean's jt_isim_from_sum overflows uint64 once column sums pass ~4.3e9 (sum^2 > 2^64);
    # at N = 12.16B it returned 0.2024 instead of 0.1209
    c = [int(v) for v in sums]
    same = sum(v * (v - 1) // 2 for v in c); diff = sum(v * (n - v) for v in c)
    value = same / (same + diff)
    bdir = Path(str(shards).replace("shards", "buckets"))
    rows = read_bucket(bdir, 0)[0][:1_000_000]  # bucket 0 is a uniform random subset; its first 1M after shuffle too
    packed_u8 = rows.view(np.uint8)  # little-endian uint32 words -> bytes in bblean's packbits(little) layout
    sigma = float(S.estimate_jt_std(packed_u8, n_samples=50))
    r = dict(n=n, isim=value, sigma_n50=sigma, k=k, threshold=value + k * sigma)
    Path(out).write_text(json.dumps(r, indent=1) + "\n"); print(r)


def check(bdir):
    commit = json.loads((Path(bdir) / "commit.json").read_text())
    rows, origin = read_bucket(bdir, 0)
    shards = Path(str(bdir).replace("buckets", "shards"))
    rng = np.random.default_rng(0); bad = 0
    for i in rng.integers(0, len(rows), 200):
        fi, ln = int(origin[i] >> 32), int(origin[i] & 0xFFFFFFFF)
        s = shards / f"{fi:05d}"
        if not Path(f"{s}.cnt").exists():
            continue
        cnt = np.fromfile(f"{s}.cnt", np.uint8); idx = np.fromfile(f"{s}.idx", np.uint16)
        line = np.fromfile(f"{s}.line", np.uint32); j = int(np.searchsorted(line, ln))
        st = int(cnt[:j].astype(np.int64).sum()); bits = idx[st:st + int(cnt[j])]
        bad += not np.array_equal(densify(np.array([cnt[j]], np.uint8), bits)[0], rows[i])
    print(f"bucket 0: {len(rows):,} rows, sampled 200 checked against shards, mismatches {bad}; commit {len(commit['done'])} shards")


if __name__ == "__main__":
    cmd = sys.argv[1]
    flags = sys.argv[2:]
    get = lambda name, d: type(d)(flags[flags.index(name) + 1]) if name in flags else d
    if cmd == "route":
        route(Path(flags[0]), Path(flags[1]), get("--buckets", 64), get("--seed", 0), "--delete-shards" in flags)
    elif cmd == "isim":
        isim(Path(flags[0]), flags[1], get("--k", 3.5))
    elif cmd == "check":
        check(Path(flags[0]))
