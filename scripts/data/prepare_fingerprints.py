"""Stream a SMILES table into packed ECFP4 fingerprints (2048 bits) for clustering.

    python scripts/data/prepare_fingerprints.py INPUT.tsv[.gz] --output-dir DIR [--id-column ID] [--smiles-column S]
        [--exclude-ids IDS.txt] [--workers N]

Used for ChEMBL34, SCUBIDOO, Enamine REAL 10M and ZINC20-ML; the exact commands are in docs/DATA.md.

Input: a (optionally gzipped) tab-separated file with a header, an ID column,
and a SMILES column. Output, in --output-dir:

* ``ids.txt`` and ``smiles.txt``: one line per kept molecule, row-aligned.
* ``ecfp4_2048.npy``: shape (N, 256) uint8. Bit ``i`` of the RDKit bit vector is
  stored in byte ``i // 8`` at bit position ``i % 8`` (little bit order), so
  ``array.view('<u4')`` gives nvMolKit's packed layout: bit ``i`` in word
  ``i // 32`` at position ``i % 32``. Every bit position is treated
  identically by Tanimoto and iSIM, so any consistent packing gives the same
  clustering.
* ``failures.tsv``: ID, SMILES, and reason for every molecule not kept.
* ``metadata.json``: counts, parameters, library versions, and SHA-256 sums.

Molecules are parsed with RDKit defaults (sanitized); no salt stripping or
other standardization is applied, matching "corrupted SMILES were discarded".
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

FP_BITS = 2048
FP_BYTES = FP_BITS // 8
RADIUS = 2  # ECFP4


def _fingerprint_chunk(rows):
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_BITS)
    kept, packed, failed = [], [], []
    for identifier, smiles in rows:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            failed.append((identifier, smiles, "rdkit_parse_failed"))
            continue
        bits = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
        packed.append(np.packbits(bits, bitorder="little"))
        kept.append((identifier, smiles))
    array = np.stack(packed) if packed else np.empty((0, FP_BYTES), dtype=np.uint8)
    return kept, array, failed


def _read_rows(path, id_column, smiles_column, exclude, chunk_size):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        id_index, smiles_index = header.index(id_column), header.index(smiles_column)
        chunk, excluded = [], []
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            identifier, smiles = fields[id_index], fields[smiles_index]
            if identifier in exclude:
                excluded.append((identifier, smiles, "excluded_by_list"))
                continue
            if not smiles:
                excluded.append((identifier, smiles, "empty_smiles"))
                continue
            chunk.append((identifier, smiles))
            if len(chunk) == chunk_size:
                yield chunk, excluded
                chunk, excluded = [], []
        if chunk or excluded:
            yield chunk, excluded


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default="chembl_id")
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--exclude-ids", type=Path, help="file with one ID per line to drop before parsing")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    started = time.time()
    exclude = set(args.exclude_ids.read_text().split()) if args.exclude_ids else set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_fp = args.output_dir / "ecfp4_2048.u8.tmp"
    counts = {"input_rows": 0, "kept": 0, "failed_parse": 0, "excluded": 0}
    with (
        open(args.output_dir / "ids.txt", "w") as ids,
        open(args.output_dir / "smiles.txt", "w") as smiles_out,
        open(args.output_dir / "failures.tsv", "w") as failures,
        open(raw_fp, "wb") as fingerprints,
        multiprocessing.get_context("spawn").Pool(args.workers) as pool,
    ):
        failures.write("id\tsmiles\treason\n")
        pending = []
        for rows, excluded in _read_rows(args.input, args.id_column, args.smiles_column, exclude, args.chunk_size):
            counts["input_rows"] += len(rows) + len(excluded)
            counts["excluded"] += len(excluded)
            for record in excluded:
                failures.write("\t".join(record) + "\n")
            pending.append(pool.apply_async(_fingerprint_chunk, (rows,)))
            # Keep chunk order while bounding in-flight work.
            while pending and (len(pending) > 2 * args.workers or pending[0].ready()):
                kept, packed, failed = pending.pop(0).get()
                for identifier, smiles in kept:
                    ids.write(identifier + "\n")
                    smiles_out.write(smiles + "\n")
                fingerprints.write(packed.tobytes())
                for record in failed:
                    failures.write("\t".join(record) + "\n")
                counts["kept"] += len(kept)
                counts["failed_parse"] += len(failed)
            print(f"read {counts['input_rows']:,} rows, kept {counts['kept']:,}", flush=True)
        for result in pending:
            kept, packed, failed = result.get()
            for identifier, smiles in kept:
                ids.write(identifier + "\n")
                smiles_out.write(smiles + "\n")
            fingerprints.write(packed.tobytes())
            for record in failed:
                failures.write("\t".join(record) + "\n")
            counts["kept"] += len(kept)
            counts["failed_parse"] += len(failed)

    # Wrap the raw bytes in an .npy header without loading them into memory.
    final = args.output_dir / "ecfp4_2048.npy"
    array = np.lib.format.open_memmap(final, mode="w+", dtype=np.uint8, shape=(counts["kept"], FP_BYTES))
    source = np.memmap(raw_fp, dtype=np.uint8, mode="r", shape=(counts["kept"], FP_BYTES))
    for begin in range(0, counts["kept"], 1_000_000):
        array[begin : begin + 1_000_000] = source[begin : begin + 1_000_000]
    array.flush()
    del array, source
    raw_fp.unlink()

    outputs = ["ids.txt", "smiles.txt", "ecfp4_2048.npy", "failures.tsv"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"path": str(args.input), "sha256": _sha256(args.input)},
        "columns": {"id": args.id_column, "smiles": args.smiles_column},
        "exclude_ids": str(args.exclude_ids) if args.exclude_ids else None,
        "counts": counts,
        "fingerprint": {
            "type": "ECFP4 (RDKit Morgan, radius 2, count simulation off, chirality off)",
            "generator": "rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)",
            "bits": FP_BITS,
            "storage": "uint8 (N, 256), numpy packbits bitorder='little'; view('<u4') is nvMolKit's layout",
        },
        "parsing": "Chem.MolFromSmiles with default sanitization; no standardization or salt stripping",
        "software": {"rdkit": rdkit.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "sha256": {name: _sha256(args.output_dir / name) for name in outputs},
        "seconds": round(time.time() - started, 1),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
