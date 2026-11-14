"""Build the datasets that are not a plain SMILES table -> ECFP4 run of prepare_fingerprints.py.

Every output directory follows the same layout as prepare_fingerprints.py: row-aligned ids.txt / smiles.txt and a
fingerprint .npy of shape (N, 256) uint8 (2048 bits, numpy.packbits(bitorder="little")). Run prepare_scaffolds.py on
each output directory afterwards to add murcko_scaffold_ids.npy.

Subcommands
-----------
subset PARENT_DIR OUT_DIR SIZE [--seed 1234]
    Random rows of PARENT_DIR/ecfp4_2048.npy (with its ids.txt, smiles.txt and, if present, murcko_scaffold_ids.npy):
    rows = sort(numpy.random.default_rng(seed).choice(N, SIZE, replace=False)). Writes rows_in_parent.npy too.
    Used for the Enamine REAL 1M sample (parent = the 10.35M Enamine set, SIZE 1000000, seed 1234).

chembl35-new CHEMBL35_SDF CHEMBL34_DIR INORGANIC_IDS OUT_DIR [--workers 64]
    ChEMBL35 molecules whose ChEMBL ID is not in ChEMBL34. CHEMBL35_SDF is the uncompressed chembl_35.sdf from EBI.
    IDs removed: every ID in CHEMBL34_DIR/ids.txt and CHEMBL34_DIR/failures.tsv (the ChEMBL34 parse failures and
    inorganic exclusions), and every ID in INORGANIC_IDS. SMILES = RDKit canonical SMILES of each SDF mol block
    (records RDKit cannot read are dropped). Fingerprint: ECFP4 2048 (ecfp4_2048.npy). Writes counts.txt.

rdkit-path SMILES_FILE OUT_DIR [--prefix np] [--csv] [--workers 64]
    RDKit path fingerprints (Chem.RDKFingerprint defaults: 2048 bits, maxPath 7) into rdkit_2048.npy, for SMILES
    that RDKit parses (others are dropped). SMILES_FILE is one SMILES per line (first whitespace field), or with
    --csv a CSV whose first column is SMILES (header skipped; MoleculeACE layout). IDs are PREFIX0, PREFIX1, ...
    Used for the ChEMBL33 natural products (prefix np) and each MoleculeACE set (prefix <set name>_, --csv).

smi-to-tsv SMI_FILE OUT_TSV [--prefix scu]
    "SMILES [name]" lines -> "id<TAB>smiles" table with IDs PREFIX0, PREFIX1, ... (blank lines skipped), the input
    for prepare_fingerprints.py --id-column id --smiles-column smiles. Used for SCUBIDOO.
"""
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator


def _ecfp(smiles):
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.stack([np.packbits(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.uint8), bitorder="little")
                     for s in smiles])


def _rdkit_path(smiles):
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        fp = Chem.RDKFingerprint(Chem.MolFromSmiles(s))
        bits = np.zeros(fp.GetNumBits(), dtype=np.uint8)
        bits[list(fp.GetOnBits())] = 1
        out.append(np.packbits(bits, bitorder="little"))
    return np.stack(out)


def _fingerprint(smiles, fn, workers):
    chunks = [smiles[i:i + 5000] for i in range(0, len(smiles), 5000)]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return np.concatenate(pool.map(fn, chunks))


def _sdf_records(chunk):
    RDLogger.DisableLog("rdApp.*")
    out = []
    for block in chunk:
        mol = Chem.MolFromMolBlock(block)
        out.append(None if mol is None else Chem.MolToSmiles(mol))
    return out


def subset(parent, out, size, seed):
    out.mkdir(parents=True, exist_ok=True)
    fps = np.load(parent / "ecfp4_2048.npy", mmap_mode="r")
    rows = np.sort(np.random.default_rng(seed).choice(len(fps), size=size, replace=False))
    np.save(out / "rows_in_parent.npy", rows)
    for f in ("ids.txt", "smiles.txt"):
        lines = (parent / f).read_text().splitlines()
        (out / f).write_text("\n".join(lines[r] for r in rows) + "\n")
    if (parent / "murcko_scaffold_ids.npy").exists():
        np.save(out / "murcko_scaffold_ids.npy", np.load(parent / "murcko_scaffold_ids.npy")[rows])
    np.save(out / "ecfp4_2048.npy", np.ascontiguousarray(fps[rows]))
    print(f"subset: {size:,} of {len(fps):,} rows -> {out}")


def chembl35_new(sdf, chembl34, inorganic_file, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    old = set((chembl34 / "ids.txt").read_text().split())
    old |= {line.split("\t")[0] for line in (chembl34 / "failures.tsv").read_text().splitlines()[1:]}
    inorganic = set(inorganic_file.read_text().split())
    ids, blocks, counts = [], [], {"records": 0, "in_chembl34": 0, "inorganic": 0}
    record = []
    with open(sdf) as handle:
        for line in handle:
            if line.startswith("$$$$"):
                counts["records"] += 1
                text = "".join(record)
                identifier = text.split("> <chembl_id>\n", 1)[1].split("\n", 1)[0].strip()
                if identifier in old:
                    counts["in_chembl34"] += 1
                elif identifier in inorganic:
                    counts["inorganic"] += 1
                else:
                    ids.append(identifier)
                    blocks.append(text.split("M  END", 1)[0] + "M  END\n")
                record = []
            else:
                record.append(line)
    chunks = [blocks[i:i + 2000] for i in range(0, len(blocks), 2000)]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        smiles = [s for part in pool.map(_sdf_records, chunks) for s in part]
    keep = [i for i, s in enumerate(smiles) if s]
    counts.update(new=len(ids), failed_parse=len(ids) - len(keep), kept=len(keep))
    ids, smiles = [ids[i] for i in keep], [smiles[i] for i in keep]
    (out / "ids.txt").write_text("\n".join(ids) + "\n")
    (out / "smiles.txt").write_text("\n".join(smiles) + "\n")
    (out / "counts.txt").write_text(str(counts) + "\n")
    np.save(out / "ecfp4_2048.npy", _fingerprint(smiles, _ecfp, workers))
    print("chembl35-new", counts)


def rdkit_path(src, out, prefix, csv, workers):
    out.mkdir(parents=True, exist_ok=True)
    lines = src.read_text().splitlines()
    if csv:
        smiles = [l.split(",")[0] for l in lines[1:] if l.strip()]
    else:
        smiles = [l.strip() for l in lines if l.strip()]
    ok = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    (out / "ids.txt").write_text("\n".join(f"{prefix}{i}" for i in range(len(ok))) + "\n")
    (out / "smiles.txt").write_text("\n".join(ok) + "\n")
    (out / "counts.txt").write_text(f"input {len(smiles)}, kept {len(ok)}\n")
    np.save(out / "rdkit_2048.npy", _fingerprint(ok, _rdkit_path, workers))
    print(f"rdkit-path: input {len(smiles)}, kept {len(ok)} -> {out}")


def smi_to_tsv(src, out, prefix):
    lines = src.read_text().splitlines()
    out.write_text("id\tsmiles\n" + "".join(f"{prefix}{i}\t{s.split()[0]}\n" for i, s in enumerate(lines) if s.strip()))
    print(f"wrote {out}")


def main():
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("subset")
    p.add_argument("parent", type=Path); p.add_argument("out", type=Path); p.add_argument("size", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p = sub.add_parser("chembl35-new")
    p.add_argument("sdf", type=Path); p.add_argument("chembl34", type=Path); p.add_argument("inorganic", type=Path)
    p.add_argument("out", type=Path); p.add_argument("--workers", type=int, default=64)
    p = sub.add_parser("rdkit-path")
    p.add_argument("src", type=Path); p.add_argument("out", type=Path); p.add_argument("--prefix", default="np")
    p.add_argument("--csv", action="store_true"); p.add_argument("--workers", type=int, default=64)
    p = sub.add_parser("smi-to-tsv")
    p.add_argument("src", type=Path); p.add_argument("out", type=Path); p.add_argument("--prefix", default="scu")
    a = ap.parse_args()
    if a.cmd == "subset":
        subset(a.parent, a.out, a.size, a.seed)
    elif a.cmd == "chembl35-new":
        chembl35_new(a.sdf, a.chembl34, a.inorganic, a.out, a.workers)
    elif a.cmd == "rdkit-path":
        rdkit_path(a.src, a.out, a.prefix, a.csv, a.workers)
    else:
        smi_to_tsv(a.src, a.out, a.prefix)


if __name__ == "__main__":
    main()
