"""Active (molecule, target) pairs from the ChEMBL35 SQLite database, for the "targets" quality measure.

    python scripts/data/chembl_actives.py CHEMBL_35_DB OUT.tsv.gz

CHEMBL_35_DB is chembl_35.db from EBI's chembl_35_sqlite.tar.gz
(https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/). Output: gzipped TSV with columns
chembl_id, target_chembl_id, one row per distinct pair. A pair is kept when ChEMBL has an activity with
pChEMBL >= 6, an exact ("=") relation, no data-validity comment, not flagged as a potential duplicate, from an assay
with confidence score >= 8 against a SINGLE PROTEIN target. Point the BB_ACTIVES config key at OUT.tsv.gz.
"""
import sqlite3
import sys

import pandas as pd

QUERY = """
SELECT DISTINCT md.chembl_id AS chembl_id, td.chembl_id AS target_chembl_id
FROM activities act
JOIN assays a ON a.assay_id = act.assay_id
JOIN target_dictionary td ON td.tid = a.tid
JOIN molecule_dictionary md ON md.molregno = act.molregno
WHERE act.pchembl_value >= 6.0
  AND act.standard_relation = '='
  AND act.data_validity_comment IS NULL
  AND act.potential_duplicate = 0
  AND a.confidence_score >= 8
  AND td.target_type = 'SINGLE PROTEIN'
"""

db, out = sys.argv[1], sys.argv[2]
frame = pd.read_sql_query(QUERY, sqlite3.connect(db))
frame.to_csv(out, sep="\t", index=False)
print(f"{len(frame):,} pairs, {frame.chembl_id.nunique():,} molecules, {frame.target_chembl_id.nunique():,} targets")
