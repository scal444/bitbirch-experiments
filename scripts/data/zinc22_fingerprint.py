"""ZINC22 2D: RDKit ECFP4 (2048 bits) for every downloaded .smi.gz file, written as sparse per-file shards.

usage: zinc22_fingerprint.py ROOT SHARD_DIR WORKERS URLS.txt [--readers 6] [--block 200000] [--poll] [--only N]

ROOT is the download directory used with download.sh and URLS.txt (urls_zinc22_2d_h04_h24.txt): file i of the URL
list is ROOT/<relative path in column 2> and becomes shard number i. For each file the shard holds
  NNNNN.cnt (uint8 set bits per molecule), NNNNN.idx (uint16 set-bit positions, concatenated),
  NNNNN.line (uint32 0-based line number in the decompressed file), NNNNN.colsum (uint64 per-bit counts),
  NNNNN.fail.tsv (line, id, SMILES, reason for molecules RDKit cannot parse), NNNNN.done (JSON summary).
Header lines and lines with fewer than two fields are skipped. Fingerprints:
rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048), RDKit default parsing.

READERS processes stream `gzip -dc` of whole files (largest first) in blocks of BLOCK lines; WORKERS parse and
fingerprint the blocks; the main process joins a file's blocks in order once all are done. Files with a .done are
skipped, and finished blocks in SHARD_DIR/.blocks are reused, so a rerun with the same --block resumes. --poll keeps
waiting for files that are still downloading. Next step: zinc22_buckets.py route.
"""
import json, multiprocessing as mp, os, subprocess, sys, time
from pathlib import Path
import numpy as np


def reader(files, blocks, results, block, tmp):
    while True:
        item = files.get()
        if item is None:
            return
        index, gz = item
        proc = subprocess.Popen(["gzip", "-dc", str(gz)], stdout=subprocess.PIPE, bufsize=1 << 24)
        b, start, buf = 0, 0, []
        for line in proc.stdout:
            buf.append(line)
            if len(buf) == block:
                put(blocks, results, tmp, index, b, start, buf); b += 1; start += len(buf); buf = []
        if proc.wait() != 0:
            raise RuntimeError(f"gzip failed on {gz}")
        if buf:
            put(blocks, results, tmp, index, b, start, buf); b += 1
        blocks.put((index, -1, b, None))  # end marker: b blocks in this file


def put(blocks, results, tmp, index, b, start, buf):
    # a block whose .fail exists was finished by an earlier run (.fail is written last, atomically): reuse it
    stem = tmp / f"{index:05d}_{b:06d}"
    if Path(f"{stem}.fail").exists():
        results.put((index, b, os.path.getsize(f"{stem}.cnt")))
    else:
        blocks.put((index, b, start, buf))


def worker(blocks, results, tmp):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from itertools import chain
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    while True:
        item = blocks.get()
        if item is None:
            return
        index, b, start, lines = item
        if b < 0:
            results.put((index, -1, start)); continue
        mols, lns, fails = [], [], []
        for k, raw in enumerate(lines):
            parts = raw.split()
            if len(parts) < 2 or parts[0].lower() == b"smiles":
                continue
            m = Chem.MolFromSmiles(parts[0].decode())
            if m is None:
                fails.append(f"{start + k}\t{parts[1].decode()}\t{parts[0].decode()}\trdkit_parse_failed\n"); continue
            mols.append(m); lns.append(start + k)
        on = [fp.GetOnBits() for fp in gen.GetFingerprints(mols, numThreads=1)]
        cnt = np.fromiter(map(len, on), np.int64, len(on))
        assert (cnt < 256).all()
        idx = np.fromiter(chain.from_iterable(on), np.uint16, int(cnt.sum()))
        stem = tmp / f"{index:05d}_{b:06d}"
        cnt.astype(np.uint8).tofile(f"{stem}.cnt"); idx.tofile(f"{stem}.idx")
        np.asarray(lns, np.uint32).tofile(f"{stem}.line")
        Path(f"{stem}.fail.tmp").write_text("".join(fails)); os.replace(f"{stem}.fail.tmp", f"{stem}.fail")
        results.put((index, b, len(mols)))


def assemble(index, nblocks, gz, out, tmp, secs):
    """Append the file's blocks in order to the shard files; memory is one block at a time."""
    stem = out / f"{index:05d}"
    sufs = (".cnt", ".idx", ".line", ".fail.tsv")
    handles = {suf: open(f"{stem}{suf}.tmp", "wb") for suf in sufs}
    colsum = np.zeros(2048, np.uint64)
    kept = set_bits = failed = 0
    for b in range(nblocks):
        p = tmp / f"{index:05d}_{b:06d}"
        cnt = np.fromfile(f"{p}.cnt", np.uint8); idx = np.fromfile(f"{p}.idx", np.uint16)
        handles[".cnt"].write(cnt.tobytes()); handles[".idx"].write(idx.tobytes())
        handles[".line"].write(Path(f"{p}.line").read_bytes())
        fail = Path(f"{p}.fail").read_bytes(); handles[".fail.tsv"].write(fail)
        colsum += np.bincount(idx, minlength=2048).astype(np.uint64)
        kept += len(cnt); set_bits += int(cnt.sum(dtype=np.int64)); failed += fail.count(b"\n")
    for suf, h in handles.items():
        h.close(); os.replace(f"{stem}{suf}.tmp", f"{stem}{suf}")
    colsum.tofile(f"{stem}.colsum.tmp"); os.replace(f"{stem}.colsum.tmp", f"{stem}.colsum")
    info = dict(file_index=index, file=str(gz), kept=kept, failed=failed, set_bits=set_bits, secs=secs, ingest="v2")
    Path(f"{stem}.done.tmp").write_text(json.dumps(info) + "\n"); os.replace(f"{stem}.done.tmp", f"{stem}.done")
    for b in range(nblocks):
        for suf in (".cnt", ".idx", ".line", ".fail"):
            os.remove(tmp / f"{index:05d}_{b:06d}{suf}")
    return info


def main():
    args = sys.argv[1:]
    opt = lambda name, d: int(args[args.index(name) + 1]) if name in args else d
    raw, out, workers, urls = Path(args[0]), Path(args[1]), int(args[2]), Path(args[3])
    readers, block, only, poll = opt("--readers", 6), opt("--block", 200000), opt("--only", -1), "--poll" in args
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".blocks"; tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.tmp"):  # half-written pieces; finished blocks (with .fail) are reused by the readers
        f.unlink()
    rels = [l.split()[1] for l in urls.read_text().splitlines() if l.strip()]
    todo = {i: raw / r for i, r in enumerate(rels) if not (out / f"{i:05d}.done").exists()}
    if only >= 0:
        todo = {only: todo[only]} if only in todo else {}
    ctx = mp.get_context("spawn")
    files, blocks, results = ctx.Queue(), ctx.Queue(maxsize=2 * workers), ctx.Queue()
    rs = [ctx.Process(target=reader, args=(files, blocks, results, block, tmp)) for _ in range(readers)]
    ws = [ctx.Process(target=worker, args=(blocks, results, tmp)) for _ in range(workers)]
    for p in rs + ws:
        p.start()
    queued, pending, t0, kept, done_n = set(), {}, time.time(), 0, 0
    started = {}

    def enqueue():
        # only files download.sh has finished (renamed from .part); largest first
        ready = [(i, p) for i, p in todo.items() if i not in queued and p.exists()]
        for i, p in sorted(ready, key=lambda r: -r[1].stat().st_size):
            files.put((i, p)); queued.add(i); started[i] = time.time()

    enqueue()
    last_poll = time.time()
    while todo:
        if not poll and not (queued & todo.keys()):
            break
        try:
            index, b, n = results.get(timeout=30)
        except Exception:
            index = None
        if poll and time.time() - last_poll > 60:
            enqueue(); last_poll = time.time()
        if index is None:
            if not poll and len(queued & set(todo)) == 0:
                break
            continue
        st = pending.setdefault(index, {"got": 0, "total": None, "mols": 0})
        if b < 0:
            st["total"] = n
        else:
            st["got"] += 1; st["mols"] += n
        if st["total"] is not None and st["got"] == st["total"]:
            info = assemble(index, st["total"], todo[index], out, tmp, time.time() - started[index])
            todo.pop(index); pending.pop(index); done_n += 1; kept += info["kept"]
            el = time.time() - t0
            print(f"{done_n} files this run ({len(todo)} left), {kept:,} mols, {el:.0f}s, {kept / max(el, 1):,.0f} mol/s",
                  flush=True)
    for _ in rs:
        files.put(None)
    for _ in ws:
        blocks.put(None)
    for p in rs + ws:
        p.join()
    print("DONE" if not todo else "INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
