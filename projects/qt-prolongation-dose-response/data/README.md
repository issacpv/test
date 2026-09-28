# Data acquisition — QT-Dose

Nothing in this folder is committed (see `.gitignore`). All patient data are credentialed.

## 1. Credentialing (required)

MIMIC-IV and MIMIC-IV-ECG are PhysioNet **credentialed** resources:

1. Create a PhysioNet account and complete the CITI "Data or Specimens Only Research" training.
2. Get credentialed on PhysioNet, then sign the data-use agreement for:
   - MIMIC-IV v3.1 — https://physionet.org/content/mimiciv/3.1/
   - MIMIC-IV-ECG v1.0 — https://physionet.org/content/mimic-iv-ecg/1.0/

Export credentials to the environment (never commit them):

```bash
export PHYSIONET_USERNAME="your_user"
export PHYSIONET_PASSWORD="your_pass"
```

## 2. What to download

The core analysis only needs the machine-measured intervals plus the medication/lab tables (a few GB). Waveforms are optional (re-delineation sensitivity analysis, ~90 GB).

Minimum (tabular):
- `mimic-iv-ecg/1.0/record_list.csv` — subject_id, study_id, ecg_time, path
- `mimic-iv-ecg/1.0/machine_measurements.csv` — machine QT, RR, QTc, axes, report fields
- `mimiciv/3.1/hosp/emar.csv.gz`, `emar_detail.csv.gz` — administration events + dose
- `mimiciv/3.1/hosp/prescriptions.csv.gz`, `pharmacy.csv.gz`
- `mimiciv/3.1/hosp/labevents.csv.gz`, `d_labitems.csv.gz` — K (50971/52610), Mg (50960), Ca (50893)
- `mimiciv/3.1/hosp/patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`
- `mimiciv/3.1/icu/inputevents.csv.gz` — continuous drug infusions (e.g., amiodarone)

Optional (waveforms): `mimic-iv-ecg/1.0/files/...` WFDB `.hea`/`.dat` for the exposure-linked subset only.

## 3. Download

```bash
# tabular tables (fast)
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --tables
python scripts/download_data.py --dataset mimiciv-hosp --out data/mimiciv-hosp --tables
python scripts/download_data.py --dataset mimiciv-icu  --out data/mimiciv-icu  --tables

# a small waveform sample for the delineation code path
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --sample

# CredibleMeds risk list: download manually after free registration at crediblemeds.org
# and save as data/external/crediblemeds_qtdrugs.csv (columns: drug, risk_category)

# FAERS torsades counts (open):
python scripts/download_data.py --dataset faers --out data/external
```

## 4. Expected layout

```
data/
  mimic-iv-ecg/
    record_list.csv
    machine_measurements.csv
    files/...                # optional WFDB waveforms (sample or full)
  mimiciv-hosp/
    emar.csv.gz  emar_detail.csv.gz  prescriptions.csv.gz  pharmacy.csv.gz
    labevents.csv.gz  d_labitems.csv.gz  patients.csv.gz  admissions.csv.gz
    diagnoses_icd.csv.gz
  mimiciv-icu/
    inputevents.csv.gz
  external/
    crediblemeds_qtdrugs.csv
    faers_tdp_counts.json
```

Notes: PhysioNet serves files over HTTPS with Basic auth; `wget -r -N -c -np --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD"` is an alternative to the Python downloader for the full waveform tree.
