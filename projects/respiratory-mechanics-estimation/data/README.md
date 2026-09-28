# Data acquisition

Expected layout (nothing here is committed):

```
data/
  mimic-iv-demo/2.2/icu/{chartevents,d_items,icustays,procedureevents}.csv.gz     # open
  eicu-crd-demo/2.0.1/{respiratoryCharting,respiratoryCare,patient}.csv.gz        # open
  kaggle-ventilator/{train.csv,test.csv}                                          # artificial lung, R/C labels
  mimiciv/3.1/icu/{chartevents,d_items,icustays,procedureevents}.csv.gz           # credentialed
  mimiciv/3.1/hosp/{admissions,patients}.csv.gz
  eicu-crd/2.0/{respiratoryCharting,respiratoryCare,patient,apachePatientResult}.csv.gz
  hirid/1.1.1/{hirid_variable_reference.csv, raw_stage/, imputed_stage/, reference_data/}
  derived/
    charting_long.parquet     # stay_id, database, time, canonical_var, value
    mechanics.parquet         # stay_id, time, E_mean, E_sd, R_mean, R_sd, dp_inferred, ...
```

## 1. Open demo databases (no credentials)

```
python scripts/download_data.py --demo
```

Fetches the MIMIC-IV demo (v2.2, 100 patients) ICU tables and the eICU-CRD demo (v2.0.1) respiratory tables from `https://physionet.org/files/<project>/<version>/`. Use them to develop label mapping, pivoting and the estimator end-to-end.

## 2. Artificial-lung ground truth (Kaggle)

1. Create a Kaggle account, accept the rules of "Google Brain - Ventilator Pressure Prediction", and place your API token at `~/.kaggle/kaggle.json`.
2. `python scripts/download_data.py --kaggle` (runs `kaggle competitions download -c ventilator-pressure-prediction` and unzips into `data/kaggle-ventilator/`).

Columns: `breath_id, R, C, time_step, u_in (inspiratory valve %), u_out (expiratory valve 0/1), pressure (cmH2O)`; R in cmH2O/L/s and C in mL/cmH2O.

## 3. Credentialed PhysioNet databases

Requirements: PhysioNet account, CITI "Data or Specimens Only Research" course, signed DUA per project. Then:

```
export PHYSIONET_USERNAME=...
export PHYSIONET_PASSWORD=...
python scripts/download_data.py --mimic     # icu/chartevents, d_items, icustays, procedureevents; hosp/admissions, patients
python scripts/download_data.py --eicu      # respiratoryCharting, respiratoryCare, patient, apachePatientResult
python scripts/download_data.py --hirid     # variable reference + raw_stage observation tables
```

Equivalent wget (MIMIC-IV example):

```
wget -N -c --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" -P data/mimiciv/3.1/icu \
  https://physionet.org/files/mimiciv/3.1/icu/chartevents.csv.gz
```

HiRID: after download, `hirid_variable_reference.csv` maps `variableid` to names; ventilator variables are found with `resp_mech.charting.map_labels` on the `Variable Name` column. The `raw_stage/observation_tables` are Parquet partitions readable with pyarrow/DuckDB.

## 4. Extraction to the harmonised long table

```
python -c "from resp_mech.charting import map_labels; ..."   # see README Methods step 2
```

The long table has one row per (stay, time, canonical variable). Per-database extraction SQL lives in `scripts/` (to be added); DuckDB reads the gzipped CSVs directly (`read_csv_auto`).
