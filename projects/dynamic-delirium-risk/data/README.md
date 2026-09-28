# Data acquisition

Only this file (and, later, small value-mapping tables under `data/value_maps/`) is committed. Patient data
never is.

## 0. Credentials (MIMIC-IV, eICU-CRD)

1. PhysioNet account: https://physionet.org/register/
2. CITI "Data or Specimens Only Research" training, uploaded at https://physionet.org/settings/credentialing/
3. Sign the DUA on https://physionet.org/content/mimiciv/3.1/ and https://physionet.org/content/eicu-crd/2.0/
4. Export in your shell only:

```bash
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"
```

## 1. Smoke test without credentials

```bash
python scripts/download_data.py --sample     # MIMIC-IV demo v2.2 (open) -> data/raw/mimic-iv-demo/2.2/
```

## 2. Full downloads

```bash
python scripts/download_data.py --db mimiciv --tables-only   # only the tables listed below
python scripts/download_data.py --db eicu --tables-only
```

Manual equivalents:

```bash
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/mimiciv/3.1/ -P data/raw/
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/eicu-crd/2.0/ -P data/raw/
```

Tables used:

| Source | Table | Use |
|---|---|---|
| MIMIC-IV icu | `d_items`, `chartevents` | CAM-ICU items, "Delirium assessment", "Richmond-RAS Scale", GCS, vitals, RASS target |
| MIMIC-IV icu | `inputevents` | sedatives / opioids / vasopressors (drug, rate, amount, patientweight) |
| MIMIC-IV icu | `procedureevents`, `icustays` | ventilation, stay times |
| MIMIC-IV hosp | `patients`, `admissions`, `labevents`, `d_labitems` | age, sex, race, language, death, labs |
| eICU-CRD | `nurseCharting` | `nursingchartcelltypevallabel` in ("Delirium Scale/Score", "Sedation Scale/Score") with CAM-ICU / RASS values |
| eICU-CRD | `infusionDrug`, `vitalPeriodic`, `lab`, `patient`, `hospital` | sedatives, vitals, labs, hospital id |

`chartevents` is very large; always filter by item id inside DuckDB, e.g.

```sql
CREATE VIEW items AS SELECT itemid, label FROM read_csv_auto('data/raw/mimiciv/3.1/icu/d_items.csv.gz')
  WHERE regexp_matches(label, '(?i)cam-icu|delirium assessment|richmond-ras|gcs');
COPY (SELECT c.* FROM read_csv_auto('data/raw/mimiciv/3.1/icu/chartevents.csv.gz') c JOIN items USING (itemid))
  TO 'data/derived/chart_delirium.parquet';
```

## 3. Expected layout

```
data/
  README.md
  value_maps/                 # versioned mapping of charted strings -> positive/negative/uta (MIMIC, eICU)
  raw/mimic-iv-demo/2.2/{hosp,icu}/*.csv.gz
  raw/mimiciv/3.1/{hosp,icu}/*.csv.gz
  raw/eicu-crd/2.0/*.csv.gz
  derived/
    chart_delirium.parquet    # filtered chartevents
    windows.parquet           # (stay_id, window_idx, w_start, w_end, state, rass_*, n_cam, ...)
    landmark_<scheme>.parquet # model-ready rows per label scheme
```
