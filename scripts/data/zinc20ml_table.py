"""Join the ZINC20-ML SMILES chunk files into one tab-separated table for prepare_fingerprints.py.

    python scripts/data/zinc20ml_table.py RAW_DIR OUT.tsv

RAW_DIR holds the extracted ZINC20-ML archives ZINC20_smiles_chunk_1/ and ZINC20_smiles_chunk_2/ (download with
download.sh and urls_zinc20ml_chunks_1_2.txt, then `tar xzf` each archive inside RAW_DIR). Each chunk directory
contains files smiles_all_NN.txt with lines "SMILES ZINC_ID". The output has the header "id<TAB>smiles" and one line
"ZINC_ID<TAB>SMILES" per input line, taken from chunk 1 then chunk 2, files in name order, lines in file order.
That table has 101,270,160 rows; the copy used for the example results had SHA-256
e0c91619bb3408953112da0ce307c5bd87df9aea97455a271089ed799d426c0f.
"""
import sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
files = [f for chunk in ("ZINC20_smiles_chunk_1", "ZINC20_smiles_chunk_2") for f in sorted((raw / chunk).glob("smiles_all_*.txt"))]
assert files, f"no smiles_all_*.txt under {raw}/ZINC20_smiles_chunk_1 or _2"
with open(out, "w") as handle:
    handle.write("id\tsmiles\n")
    for f in files:
        with open(f) as src:
            for line in src:
                smiles, identifier = line.split()
                handle.write(f"{identifier}\t{smiles}\n")
print(f"wrote {out} from {len(files)} files")
