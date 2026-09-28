# Data acquisition

`data/` is git-ignored except for this file. Two sources: openFDA (open) and MIMIC-IV (+ MIMIC-IV-ECG),
which require PhysioNet credentialing.

## 1. openFDA drug adverse-event API (open)

Get a free API key at https://open.fda.gov/apis/authentication/ and export it:

```bash
export OPENFDA_API_KEY="..."      # optional; raises limits from 40 to 240 requests/min
```

The client (`faers_ehr.openfda_client.OpenFDAClient`) uses the `count` endpoint and `meta.results.total`
for 2x2 tables, so a full drug x outcome table needs ~4 requests per pair. Pagination of raw records
uses `skip` (<= 25,000) and then the `Link` header (`search_after`) the API returns.

Examples:

```bash
# counts of reaction terms among suspect-drug reports for amiodarone (1 request)
python scripts/download_data.py --source openfda --sample
# 2x2 tables for the ingredients in configs/ingredients.txt x the 7 outcome PT groups
python scripts/download_data.py --source openfda --ingredients configs/ingredients.txt --out data/openfda
```

Useful fields: `patient.drug.openfda.generic_name`, `patient.drug.openfda.rxcui`,
`patient.drug.medicinalproduct`, `patient.drug.drugcharacterization` (1 suspect, 2 concomitant,
3 interacting), `patient.reaction.reactionmeddrapt`, `receivedate`, `serious`,
`primarysource.qualification` (1 physician, 2 pharmacist, 3 other HCP, 4 lawyer, 5 consumer),
`patient.patientsex`, `patient.patientonsetage`, `safetyreportid`, `safetyreportversion`.

Bulk files (for deduplication and reproducibility): `https://api.fda.gov/download.json` lists the
quarterly partitions under `results.drug.event.partitions[*].file`;
`python scripts/download_data.py --source openfda-bulk --quarters 2023q1 2023q2` fetches them.

## 2. MIMIC-IV v3.1 (credentialed)

1. PhysioNet account, CITI "Data or Specimens Only Research" training, sign the MIMIC-IV DUA:
   https://physionet.org/content/mimiciv/3.1/
2. `export PHYSIONET_USER=... PHYSIONET_PASS=...`
3. `python scripts/download_data.py --source mimiciv` downloads only the tables this project uses:

```
hosp/patients.csv.gz  hosp/admissions.csv.gz  hosp/prescriptions.csv.gz  hosp/emar.csv.gz
hosp/emar_detail.csv.gz  hosp/labevents.csv.gz  hosp/d_labitems.csv.gz  hosp/diagnoses_icd.csv.gz
icu/icustays.csv.gz  icu/inputevents.csv.gz
```

`labevents.csv.gz` is ~2.5 GB compressed; DuckDB reads it in place (`read_csv_auto`).

## 3. MIMIC-IV-ECG v1.0 (open on PhysioNet; linkage needs MIMIC-IV)

```bash
python scripts/download_data.py --source mimic-iv-ecg     # machine_measurements.csv + record_list.csv only
```

`machine_measurements.csv` has `subject_id, study_id, ecg_time, rr_interval, p_onset, p_end, qrs_onset,
qrs_end, t_end, p_axis, qrs_axis, t_axis` plus `report_0..report_17` free-text machine statements
(the text may say "PROLONGED QT INTERVAL"; treat it as machine output, never send it to external APIs).

## 4. RxNorm (open)

`faers_ehr.rxnorm.RxNavClient` calls https://rxnav.nlm.nih.gov/REST/ and caches to `data/rxnorm_cache.json`.
No key is needed; keep requests under ~20/s.

## Expected layout

```
data/
  openfda/
    two_by_two.csv              # drug, outcome, a, b, c, d, n, query strings, retrieved_at
    counts_<drug>.json          # reaction counts per drug (optional)
    bulk/drug-event-2023q1-*.json.zip
  mimiciv/3.1/hosp/*.csv.gz  mimiciv/3.1/icu/*.csv.gz
  mimic-iv-ecg/1.0/machine_measurements.csv  record_list.csv
  rxnorm_cache.json
  interim/exposures.parquet   # subject_id, hadm_id, ingredient, first_admin_time, ...
  interim/outcomes_<name>.parquet
```

Set `MIMICIV_ROOT=data/mimiciv/3.1` and `MIMIC_ECG_ROOT=data/mimic-iv-ecg/1.0` for the SQL helpers.
