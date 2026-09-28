# Data acquisition

Nothing under `data/` is committed except this file and (later) the curated model library under
`data/models/` (plain JSON/CSV of published parameter values with citations; no patient data).

## 0. Credentials (MIMIC-IV, eICU-CRD)

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training and upload the certificate:
   https://physionet.org/settings/credentialing/
3. Sign the data use agreement on https://physionet.org/content/mimiciv/3.1/ (and eICU if used).
4. Export credentials in your shell only (never into files):

```bash
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"
```

## 1. Smoke test without credentials: MIMIC-IV demo

The demo (100 patients, identical schema, open access) is enough to run the whole pipeline end to end:

```bash
python scripts/download_data.py --sample            # -> data/raw/mimic-iv-demo/2.2/{hosp,icu}/
```

## 2. Full download (credentialed)

```bash
python scripts/download_data.py --full              # hosp + icu modules of MIMIC-IV v3.1
python scripts/download_data.py --full --tables-only  # only the tables this project reads (recommended)
```

Equivalent manual command:

```bash
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
     https://physionet.org/files/mimiciv/3.1/ -P data/raw/
```

Tables read by this project (all `csv.gz`):

| Module | Table | Why |
|---|---|---|
| hosp | `d_labitems`, `labevents` | vancomycin / gentamicin / tobramycin / amikacin levels, creatinine |
| hosp | `emar`, `emar_detail` | barcode administration times, dose given, infusion rate |
| hosp | `prescriptions` | scheduled orders (fallback dose source; dose/route) |
| hosp | `patients`, `admissions`, `omr` | age, sex, race, anchor_year_group, weight/height |
| icu | `d_items`, `inputevents` | vancomycin / aminoglycoside infusions with start/end/rate and patient weight |
| icu | `icustays`, `chartevents`, `procedureevents`, `outputevents` | stay times, daily weight, CRRT, urine output for KDIGO |

`labevents`, `chartevents` and `emar_detail` are large (several GB compressed). Query them with DuckDB
directly on the gz files rather than loading with pandas, e.g.

```sql
SELECT * FROM read_csv_auto('data/raw/mimiciv/3.1/hosp/labevents.csv.gz')
WHERE itemid IN (SELECT itemid FROM read_csv_auto('data/raw/mimiciv/3.1/hosp/d_labitems.csv.gz')
                 WHERE regexp_matches(lower(label), 'vancomycin|gentamicin|tobramycin|amikacin|creatinine'));
```

## 3. Expected layout

```
data/
  README.md
  models/                     # curated published popPK parameter sets (JSON) + checkers' expected values
  raw/
    mimic-iv-demo/2.2/{hosp,icu}/*.csv.gz
    mimiciv/3.1/{hosp,icu}/*.csv.gz
    eicu-crd/2.0/*.csv.gz      # optional
  derived/
    doses.parquet             # (subject_id, hadm_id, stay_id, course_id, start, end, amount_mg, source)
    levels.parquet            # (subject_id, course_id, charttime, value_mg_L, drug, trough_inferred)
    covariates.parquet        # per course/time: weight, creatinine, egfr, crcl, crrt, vasopressor, aki
```

## 4. Item ids

Item ids are resolved at run time by regex over `d_items.label` / `d_labitems.label`
(`bayes_pk.dosing_records.resolve_itemids`). The resolved table is printed and saved to
`data/derived/itemids.csv` so that the exact ids used in a run are recorded. Do not hard-code ids from memory.
