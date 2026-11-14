# Data

Where each library comes from, the commands that turn it into fingerprints, scaffolds and target annotations, and
the counts and checksums to expect. Scripts are in `scripts/data/`; run them in the `bitbirch` environment (RDKit
2025.09.6, NumPy 2.2.6). Other RDKit versions can change a few fingerprints, and with them the clusters.

The commands write under `$DATA`; set `BB_DATA` in `config/paths.env` to the same folder. The folder names are the
ones the scripts expect.

## Format

- **Fingerprints**: `.npy`, `uint8`, shape (N, 256): 2048 bits packed with `numpy.packbits(bits, bitorder="little")`
  (bit `i` in byte `i // 8`, position `i % 8`). bblean takes these rows as they are; nvMolKit gets the same memory as
  little-endian 32-bit words, `np.ascontiguousarray(fps).view("<u4")`.
- **ECFP4**: `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`, no counts, no chirality.
  **RDKit path** (natural products, MoleculeACE): `Chem.RDKFingerprint(mol)` with defaults (2048 bits, max path 7).
- **Parsing**: `Chem.MolFromSmiles`, default sanitization, no salt stripping. Failures are dropped (listed in
  `failures.tsv` where the script writes one).
- **Row-aligned files**: `ids.txt`, `smiles.txt`, and from `prepare_scaffolds.py` `murcko_scaffold_ids.npy` (`int32`
  per row; line k of `murcko_scaffolds.txt` is scaffold k; 0 = acyclic, −1 = parse failure), using
  `MurckoScaffold.MurckoScaffoldSmiles(mol=...)`. `prepare_fingerprints.py` also writes `metadata.json` with counts,
  versions and SHA-256 of inputs and outputs.

## Input orders and subsets

Order `s` of a library with N rows is `numpy.random.default_rng(s).permutation(N)`: row `order[i]` goes in i-th,
labels are stored per original row (`perm` in `scripts/common/paths.py`). Orders used: ChEMBL34, SCUBIDOO and
Enamine 1M 0–2 (ChEMBL34 also 3–9 at k = 3.5); ChEMBL35 additions and natural products 0–4; MoleculeACE and all
timing runs 0.

ZINC20 timing subsets (10M, 25M, 50M) are the first N rows of the 101M set in order 0
(`scripts/run/perf_inputs.py`). The Enamine 1M sample is a fixed random subset of the 10.35M set (seed 1234, below).

## Thresholds

t = iSIM + k·σ, iSIM the exact mean pairwise Tanimoto, σ from `bblean.similarity.estimate_jt_std(fps, n_samples=50)`.
Values for these libraries: `results_gb300_72coreARM/data/thresholds/isim_sigma.json`. Recompute with
`scripts/metrics/thresholds.py` (quality libraries and Enamine 10M), `scripts/data/isim_sigma_streamed.py`
(ZINC20 101M), and `scripts/data/zinc22_buckets.py isim` (ZINC22).

## Summary

| library (key) | used for | molecules | fingerprint | raw input | public? |
|---|---|---:|---|---|---|
| ChEMBL34 (`chembl34`) | quality | 2,409,133 | ECFP4 | EBI `chembl_34_chemreps.txt.gz` | yes |
| SCUBIDOO (`scu`) | quality | 999,794 | ECFP4 | Zenodo 17796465 `scubidoo.smi` | yes |
| Enamine REAL 1M sample (`ena1m`) | quality | 1,000,000 | ECFP4 | subset of Enamine REAL 10M | needs Enamine registration, release not recorded |
| ChEMBL35 additions (`c35`) | quality | 65,728 | ECFP4 | EBI `chembl_35.sdf.gz` + ChEMBL34 IDs | yes |
| ChEMBL33 natural products (`np`) | quality | 64,086 | RDKit path | `data/chembl_33_np.smi` in mqcomplab/bitbirch | yes |
| ChEMBL35 actives (`BB_ACTIVES`) | target measures | 919,917 pairs | — | EBI `chembl_35_sqlite.tar.gz` | yes |
| Enamine REAL 10M (`ena10m`) | timing | 10,353,526 | ECFP4 | Enamine REAL 10M subset file | needs Enamine registration, release not recorded |
| ZINC20-ML 101M (`zinc101m`) and subsets 10M / 25M / 50M | timing | 101,270,110 | ECFP4 | files.docking.org ZINC20-ML chunks 1–2 | yes |
| ZINC22 2D H04–H24 | 12-billion-molecule run | 12,156,333,129 | ECFP4 | files.docking.org ZINC22 2D | yes, with the publicly posted login |
| MoleculeACE (30 sets) | BitBIRCH-paper comparisons | 615–3,657 each, 48,714 in all | RDKit path | github.com/molML/MoleculeACE | yes |

## ChEMBL34

- Download: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_chemreps.txt.gz
  (SHA-256 `7bcdcdc81103f03ed0e2c474ddb3c8faa2714a97af6e5356664b792011a9e160`, from EBI's `checksums.txt`).
  CC BY-SA 3.0 (EMBL-EBI).
- Inorganics are removed, as in the best-practices paper. The chemreps file has no inorganic flag, so the IDs came
  from the ChEMBL API (`https://www.ebi.ac.uk/chembl/api/data/molecule.json?inorganic_flag=1`, which served ChEMBL 37
  then). The 204 IDs are in `scripts/data/chembl_inorganic_ids.txt`; 121 are in ChEMBL34.

```bash
mkdir -p $DATA/chembl34
python scripts/data/prepare_fingerprints.py chembl_34_chemreps.txt.gz --output-dir $DATA/chembl34 \
  --exclude-ids scripts/data/chembl_inorganic_ids.txt --workers 6
python scripts/data/prepare_scaffolds.py $DATA/chembl34 --workers 6
```

Counts: 2,409,270 rows in, 121 inorganic removed, 16 parse failures, **2,409,133 kept**; 712,177 unique scaffolds.
SHA-256: `ecfp4_2048.npy` `dbf802dce1881a5679583a916adbcf1b6ef2e757bcd7bbbc66a13ec95ad1e733`,
`ids.txt` `6bb30987232d0c4b34fb1460299e9910e516122dd2bb20ef87bab92d3d78634d`.

The best-practices paper used 2,764,449 ChEMBL34 molecules from ChEMBL's download tool (their set is Zenodo 17796465,
`chembl.smi`). This repo uses the release file, so cluster counts differ from the paper's.

## SCUBIDOO

- Download: `scubidoo.smi` from https://zenodo.org/records/17796465 (the best-practices paper's set). SHA-256
  `31e9e941e51c60878dcbccbfac72473431a9b1ca24452d388a14c340985db30d`.

```bash
mkdir -p $DATA/scubidoo
python scripts/data/prepare_derived.py smi-to-tsv scubidoo.smi $DATA/scubidoo/scubidoo.tsv --prefix scu
python scripts/data/prepare_fingerprints.py $DATA/scubidoo/scubidoo.tsv --output-dir $DATA/scubidoo \
  --id-column id --smiles-column smiles --workers 16
python scripts/data/prepare_scaffolds.py $DATA/scubidoo --workers 16
```

Counts: **999,794** molecules, no parse failures; 468,749 unique scaffolds. SHA-256: `scubidoo.tsv`
`ea29259fe9360ce9f40c0d527cc787dcbc34d19e012fe29ae847c2a47f92a082`, `ecfp4_2048.npy`
`54c0c3bd23a0bc0af79fa07db52f6ba469d9087e775f38eaa3b6f8658f775395`.

## Enamine REAL 10M and the 1M sample

- Source: a 10,353,526-molecule Enamine REAL subset in CXSMILES (tab-separated, columns `smiles`, `id`, then
  properties) from https://enamine.net/compound-collections/real-compounds (free registration). The release and file
  name weren't recorded. SHA-256 `71dff9e232d8606cd2d5e92f184f0fddc383efb5f1c2633ca0d567b052bb5933`. REAL gets
  updated, so a fresh download will likely differ; the comparison between implementations still works on it.

```bash
mkdir -p $DATA/enamine_real_10m
python scripts/data/prepare_fingerprints.py enamine_real_10M.cxsmiles --output-dir $DATA/enamine_real_10m \
  --id-column id --smiles-column smiles --workers 64
python scripts/data/prepare_scaffolds.py $DATA/enamine_real_10m --workers 64
python scripts/data/prepare_derived.py subset $DATA/enamine_real_10m $DATA/enamine_real_10m/subset_1m 1000000 --seed 1234
```

Counts: **10,353,526** molecules, no parse failures; 5,467,253 unique scaffolds. `ecfp4_2048.npy` SHA-256
`814c17fe85fba9247235e343617e451ffde5c7ff07d274f5103b128ddb1554ae`. The 1M sample is rows
`sort(numpy.random.default_rng(1234).choice(10353526, 1000000, replace=False))` (`rows_in_parent.npy`). Run the
scaffold step before `subset` so the sample gets scaffold IDs.

## ChEMBL35 additions

ChEMBL35 molecules whose ID isn't in ChEMBL34 (including ChEMBL34's parse failures and excluded IDs), minus the
inorganic list. SMILES are RDKit canonical SMILES of the SDF mol blocks. Needs the ChEMBL34 folder above.

- Download: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35.sdf.gz
  (EBI's SHA-256 `a53871b075443a9fca3b72410a0035d1d0eeb56639d7764aee3bca53b0580e7b`), then `gunzip`
  (uncompressed SHA-256 `a18689838eee92063372291c54e919e3489292da2254d741276f2f64ef2c43eb`).

```bash
python scripts/data/prepare_derived.py chembl35-new chembl_35.sdf $DATA/chembl34 \
  scripts/data/chembl_inorganic_ids.txt $DATA/chembl35_new --workers 64
python scripts/data/prepare_scaffolds.py $DATA/chembl35_new --workers 64
```

Counts (`counts.txt`): 2,474,590 SDF records, 2,408,861 already in ChEMBL34, 1 inorganic, no parse failures,
**65,728 kept**; 29,600 unique scaffolds.

## ChEMBL33 natural products

- Source: `data/chembl_33_np.smi` in https://github.com/mqcomplab/bitbirch (commit 311fbf7; SHA-256
  `f1e2a9b2a39e532304f37b9d28f80b49f19e3c009cc1c37d839faa8a7e0da2bd`), the natural-product set of the BitBIRCH and
  refinement papers. Used as is.

```bash
python scripts/data/prepare_derived.py rdkit-path chembl_33_np.smi $DATA/chembl33_np --prefix np
python scripts/data/prepare_scaffolds.py $DATA/chembl33_np
```

Counts: **64,086** molecules (all parse), RDKit path fingerprints in `rdkit_2048.npy`; 18,132 unique scaffolds.

## ChEMBL target annotations (`BB_ACTIVES`)

For the target measures on ChEMBL34 and ChEMBL35 additions; only molecules with exactly one target are scored.

- Download: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz
  (EBI's SHA-256 `381842aae142bf68b035f8d1aba94dfb715ad5d979640f16186b0f9d44fa5320`), then `tar xzf`.

```bash
python scripts/data/chembl_actives.py chembl_35/chembl_35_sqlite/chembl_35.db $DATA/chembl35/actives.tsv.gz
```

Kept: (molecule, target) pairs with an activity of pChEMBL ≥ 6, relation "=", no data-validity comment, not a
potential duplicate, assay confidence ≥ 8, single-protein target. **919,917** pairs (`chembl_id`,
`target_chembl_id`). Point `BB_ACTIVES` at the file.

## ZINC20-ML 101M and the timing subsets

- Download (no login): `ZINC20_smiles_chunk_1.tar.gz` and `ZINC20_smiles_chunk_2.tar.gz` from
  https://files.docking.org/zinc20-ML/smiles/ (SHA-256
  `bc3c87c22d6f4c216d4b5362ac59ce839db1f7d25906c292a29e70582f4d84ab` and
  `c30294d1797ed4217999aba528c6ed3fff06b08281e7a1bdd8177aeeb42dccf2`).

```bash
mkdir -p $DATA/zinc20ml/raw
bash scripts/data/download.sh scripts/data/urls_zinc20ml_chunks_1_2.txt $DATA/zinc20ml/raw 2
(cd $DATA/zinc20ml/raw && tar xzf ZINC20_smiles_chunk_1.tar.gz && tar xzf ZINC20_smiles_chunk_2.tar.gz)
python scripts/data/zinc20ml_table.py $DATA/zinc20ml/raw $DATA/zinc20ml/raw/zinc20ml_100m.tsv
python scripts/data/prepare_fingerprints.py $DATA/zinc20ml/raw/zinc20ml_100m.tsv --output-dir $DATA/zinc20ml \
  --id-column id --smiles-column smiles --workers 64
python scripts/data/isim_sigma_streamed.py $DATA/zinc20ml/ecfp4_2048.npy zinc101m_thresholds.json
```

Counts: joined table 101,270,160 rows (SHA-256 `e0c91619bb3408953112da0ce307c5bd87df9aea97455a271089ed799d426c0f`);
50 parse failures; **101,270,110 kept**. `ecfp4_2048.npy` SHA-256
`f2376b67fe0bafe71c09f1a5159b8b6c7d0ff69ef4c49b00731cc608303ae8ba` (26 GB). iSIM 0.12439, σ 0.03855, t at
k = 3.5 = 0.25930; the 10M, 25M and 50M subsets use the same threshold. No scaffolds needed (timing only).

## ZINC22 2D (12.16 billion molecules)

- Download: the 1,134 files of heavy-atom tiers H04–H24 from https://files.docking.org/zinc22/2d/ (list in
  `scripts/data/urls_zinc22_2d_h04_h24.txt`, about 146 GB of `.smi.gz`). The login is posted on
  https://files.docking.org/zinc22/. ZINC22 can change and no checksums were published or recorded.
- Disk: about 150 GB for the download plus several hundred GB for shards and buckets. Every step can resume.

```bash
export DOCKING_AUTH=user:password       # the credentials posted on files.docking.org/zinc22/
bash scripts/data/download.sh scripts/data/urls_zinc22_2d_h04_h24.txt $DATA/zinc22_2d 3
python scripts/data/zinc22_fingerprint.py $DATA/zinc22_2d $DATA/zinc22_2d_fp/shards 70 \
  scripts/data/urls_zinc22_2d_h04_h24.txt --readers 6 --block 200000
python scripts/data/zinc22_buckets.py route $DATA/zinc22_2d_fp/shards $DATA/zinc22_2d_fp/buckets --buckets 128 --seed 0
python scripts/data/zinc22_buckets.py isim $DATA/zinc22_2d_fp/shards $DATA/zinc22_2d_fp/threshold.json --k 3.5
```

Counts: **12,156,333,129** kept, 71 parse failures. iSIM 0.120895 (exact integers; bblean's `jt_isim_from_sum`
overflows at this size and gives a wrong 0.2024), σ 0.041615 from 1M random rows, t = 0.266549 at k = 3.5.

Input order: each row goes to bucket `numpy.random.default_rng([0, file_index]).integers(128)`, buckets go in
order 0–127, and each bucket is shuffled with `default_rng([0, 1000003, bucket]).permutation` (`read_bucket` in
`zinc22_buckets.py`). Uniformly random, but not `default_rng(0).permutation(N)`.

The ZINC22 clustering run itself isn't part of the quality and timing pipelines. The figures only read its small
summaries in `results_gb300_72coreARM/data/zinc22/`, made by
`python scripts/data/zinc22_export.py $DATA/zinc22_2d_fp METRICS_DIR`.

## MoleculeACE (30 sets)

- Source: https://github.com/molML/MoleculeACE, `MoleculeACE/Data/benchmark_data/CHEMBL*.csv` (commit 7e6de0b), the
  30 sets of the BitBIRCH paper's MoleculeACE comparison; first CSV column is SMILES.

```bash
git clone https://github.com/molML/MoleculeACE && git -C MoleculeACE checkout 7e6de0b
for f in MoleculeACE/MoleculeACE/Data/benchmark_data/CHEMBL*.csv; do
  s=$(basename "$f" .csv)
  python scripts/data/prepare_derived.py rdkit-path "$f" "$DATA/moleculeace/$s" --prefix "${s}_" --csv --workers 16
  python scripts/data/prepare_scaffolds.py "$DATA/moleculeace/$s"
done
```

Counts: 30 sets of 615–3,657 molecules, 48,714 in all, every SMILES parses (`counts.txt` per set); RDKit path
fingerprints in `rdkit_2048.npy`.
