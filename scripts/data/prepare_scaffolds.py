"""Compute Bemis-Murcko scaffolds for a dataset's smiles.txt.

    python scripts/data/prepare_scaffolds.py DATASET_DIR [--workers N]

Writes, next to smiles.txt:

* ``murcko_scaffold_ids.npy``: int32, one scaffold ID per molecule (row-aligned).
* ``murcko_scaffolds.txt``: scaffold SMILES, line k is scaffold ID k.
  ID 0 is the empty scaffold (acyclic molecules); -1 marks parse failures.

Scaffolds come from ``MurckoScaffold.MurckoScaffoldSmiles(mol=...)`` (RDKit,
plain Bemis-Murcko: ring systems plus linkers, atom and bond types kept).
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffolds(smiles_chunk):
    RDLogger.DisableLog("rdApp.*")
    result = []
    for smiles in smiles_chunk:
        molecule = Chem.MolFromSmiles(smiles)
        result.append(None if molecule is None else MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="directory containing smiles.txt")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()

    smiles = (args.dataset / "smiles.txt").read_text().splitlines()
    chunks = [smiles[i : i + args.chunk_size] for i in range(0, len(smiles), args.chunk_size)]
    vocabulary = {"": 0}
    ids = np.empty(len(smiles), dtype=np.int32)
    position = 0
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for scaffolds in pool.imap(_scaffolds, chunks):
            for scaffold in scaffolds:
                ids[position] = -1 if scaffold is None else vocabulary.setdefault(scaffold, len(vocabulary))
                position += 1
    np.save(args.dataset / "murcko_scaffold_ids.npy", ids)
    (args.dataset / "murcko_scaffolds.txt").write_text("\n".join(vocabulary) + "\n")
    summary = {
        "molecules": len(ids),
        "unique_scaffolds": len(vocabulary),
        "acyclic": int((ids == 0).sum()),
        "parse_failures": int((ids == -1).sum()),
        "rdkit": rdkit.__version__,
        "method": "MurckoScaffold.MurckoScaffoldSmiles(mol=...)",
    }
    (args.dataset / "murcko_scaffolds.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
