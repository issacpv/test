# Data acquisition

Nothing here is committed. Expected layout:

```
data/
  mimic-iv-note/note/     # credentialed: radiology.csv.gz, radiology_detail.csv.gz
  mimic-iv-ed/ed/         # credentialed: edstays, triage, diagnosis
  mimiciv/hosp/           # credentialed: patients, admissions
  mimiciv/icu/            # credentialed: icustays
  mimic-cxr-jpg/          # credentialed: mimic-cxr-2.0.0-metadata.csv.gz, mimic-cxr-2.0.0-chexpert.csv.gz (no images)
  mimic-iv-ed-demo/ed/    # open demo
  mimic-iv-demo/{hosp,icu}/
  synthetic/              # schema-matching synthetic tables written by --synthetic
  derived/                # exam table, linked stays, model inputs
```

## Credentialed PhysioNet resources

1. PhysioNet account, CITI "Data or Specimens Only Research", sign the DUA for MIMIC-IV-Note v2.2, MIMIC-IV-ED v2.2, MIMIC-IV v3.1 and MIMIC-CXR-JPG v2.1.0.
2. `export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...`
3. `python scripts/download_data.py --note --ed --hosp --cxr-labels` runs `wget -N -c --user ... --password ...` for exactly the files above. `radiology.csv.gz` is ~1 GB compressed; convert to parquet with DuckDB:

```python
import duckdb
duckdb.sql("COPY (SELECT note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text FROM read_csv_auto('data/mimic-iv-note/note/radiology.csv.gz')) TO 'data/derived/radiology.parquet'")
```

Only the MIMIC-CXR-JPG metadata and CheXpert label CSVs are needed (no DICOM/JPEG download).

## Open demos and synthetic data (no credentials)

- `python scripts/download_data.py --sample` fetches the MIMIC-IV-ED demo and MIMIC-IV demo tables.
- `python scripts/download_data.py --synthetic` writes `data/synthetic/{radiology,radiology_detail,edstays,triage}.csv` with the real column names and plausible timestamps so `src/rad_delay/linkage.py` can be run end-to-end. There is no public note demo.

## Checks

`python scripts/download_data.py --check`.
