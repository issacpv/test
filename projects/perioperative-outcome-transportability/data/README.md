# Data acquisition

Nothing here is committed. Expected layout:

```
data/
  inspire/            # PhysioNet credentialed: operations, labs, medications, diagnosis, vitals, ward_vitals (csv.gz)
  mover/              # UCI DUA: patient_information, surgery_information, labs, diagnoses, procedures ... (csv)
  mimiciv/
    hosp/             # admissions, patients, transfers, services, procedures_icd, diagnoses_icd, labevents, d_labitems
    icu/              # icustays
  mimic-iv-demo/      # open demo (100 patients) for dry runs; same layout as mimiciv/
  vitaldb/            # cases.csv, trks.csv (+ per-case files only if needed)
  derived/            # encounter tables, features, predictions
```

## 1. INSPIRE (PhysioNet, credentialed)

1. PhysioNet account + CITI "Data or Specimens Only Research" + sign the INSPIRE DUA.
2. `export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...`
3. `python scripts/download_data.py --inspire` runs
   `wget -r -N -c -np --user ... --password ... https://physionet.org/files/inspire/<version>/`.
   Check https://physionet.org/content/inspire/ for the current version and set `--inspire-version` if it changed.
4. Read the data dictionary shipped with the release; INSPIRE times are stored as offsets (minutes) relative to an anchor, so all windows in this project are computed in minutes from surgery start/end.

## 2. MOVER (UC Irvine, data-use agreement)

1. Go to https://mover.ics.uci.edu/ and sign the DUA (free for legitimate research).
2. Download the EHR tables (the waveform archives are optional here) into `data/mover/`.
3. Record the release date you downloaded in `data/mover/RELEASE.txt`; column names have changed between releases and the loaders in `src/periop_transport/cohorts.py` take an explicit column map.

## 3. MIMIC-IV v3.1 (PhysioNet, credentialed)

`python scripts/download_data.py --mimiciv` fetches only the tables this project needs:
`hosp/{admissions,patients,transfers,services,procedures_icd,diagnoses_icd,labevents,d_labitems}.csv.gz` and `icu/icustays.csv.gz`.
`labevents` is ~2 GB compressed; convert to parquet with DuckDB after download:

```python
import duckdb
duckdb.sql("COPY (SELECT * FROM read_csv_auto('data/mimiciv/hosp/labevents.csv.gz')) TO 'data/mimiciv/hosp/labevents.parquet'")
```

## 4. MIMIC-IV demo (open) for dry runs

`python scripts/download_data.py --sample` downloads the MIMIC-IV Clinical Database Demo v2.2 tables (no credentials) into `data/mimic-iv-demo/` so the full MIMIC pipeline can be run end-to-end on 100 patients.

## 5. VitalDB (open API)

`python scripts/download_data.py --vitaldb-index` stores `cases.csv` and `trks.csv` from https://api.vitaldb.net. Per-case waveforms are needed only for the intraoperative sensitivity analysis (`pip install vitaldb`).

## Checks

`python scripts/download_data.py --check` lists which parts are present.
