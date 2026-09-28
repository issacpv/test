# Data acquisition

Nothing here is committed. Expected layout:

```
data/
  mimiciv/
    icu/     # credentialed: d_items.csv.gz, icustays.csv.gz, chartevents.csv.gz (filter before loading!)
    hosp/    # credentialed: diagnoses_icd.csv.gz, admissions.csv.gz, patients.csv.gz
  mimic-iv-demo/{icu,hosp}/   # open demo, same tables, 100 patients
  derived/
    chartevents_nursing.parquet   # chartevents restricted to resolved item ids (DuckDB)
    braden.parquet, labels.parquet, landmarks.parquet
```

## MIMIC-IV v3.1 (PhysioNet, credentialed)

1. PhysioNet account, CITI "Data or Specimens Only Research", sign the MIMIC-IV DUA.
2. `export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...`
3. `python scripts/download_data.py --mimiciv` fetches `icu/d_items`, `icu/icustays`, `icu/chartevents` (~3 GB compressed), `hosp/diagnoses_icd`, `hosp/admissions`, `hosp/patients`.
4. Resolve item ids and filter `chartevents` once with DuckDB (never load the full table into pandas):

```python
import duckdb, pandas as pd
from nursing_risk.items import resolve_items
d_items = pd.read_csv("data/mimiciv/icu/d_items.csv.gz")
ids = resolve_items(d_items)["itemid"].tolist()
duckdb.sql(f"""
COPY (SELECT subject_id, hadm_id, stay_id, charttime, itemid, value, valuenum
      FROM read_csv_auto('data/mimiciv/icu/chartevents.csv.gz')
      WHERE itemid IN ({','.join(map(str, ids))}))
TO 'data/derived/chartevents_nursing.parquet'
""")
```

## MIMIC-IV demo (open)

`python scripts/download_data.py --sample` fetches the same tables from the MIMIC-IV Clinical Database Demo v2.2 (no credentials), and `python scripts/download_data.py --resolve-items` prints which Braden / skin / prevention / fall items resolve on whichever `d_items` is present — the first thing to check before any modelling.

## eICU-CRD (optional, credentialed)

Only if an external availability check is wanted: `nurseCharting.csv.gz` and `nurseAssessment.csv.gz` from https://physionet.org/content/eicu-crd/2.0/ with the same wget pattern; search `nursingchartcelltypevallabel` / `celllabel` for "Braden" and "Skin".

## Checks

`python scripts/download_data.py --check`.
