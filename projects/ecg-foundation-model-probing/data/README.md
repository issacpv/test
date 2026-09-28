# Data acquisition — ECG-Probe

Nothing here is committed. Open data (PTB-XL) needs no login; MIMIC is credentialed.

## 1. PTB-XL (open)

```bash
python scripts/download_data.py --dataset ptbxl --out data/ptbxl --sample   # metadata + 20 records
python scripts/download_data.py --dataset ptbxl --out data/ptbxl            # full (~3 GB)
```
Files used: `ptbxl_database.csv` (age, sex, scp_codes, patient_id, strat_fold), `scp_statements.csv`, and `records500/` WFDB waveforms.

## 2. MIMIC-IV-ECG + demographics (credentialed)

Complete PhysioNet CITI training and sign the DUA for MIMIC-IV v3.1 and MIMIC-IV-ECG v1.0. Export credentials (never commit):

```bash
export PHYSIONET_USERNAME="your_user"
export PHYSIONET_PASSWORD="your_pass"
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --tables
python scripts/download_data.py --dataset mimiciv-hosp --out data/mimiciv-hosp --tables
```
Used: `record_list.csv`, a waveform subset; `hosp/admissions.csv.gz` (race), `hosp/patients.csv.gz` (anchor_age, gender).

## 3. Model weights (verify licence)

Download each foundation-model checkpoint from its official repository (ECG-FM, HuBERT-ECG, ECGFounder) into `models/`. The code runs without them using a deterministic mock encoder, so you can develop the probing/erasure pipeline before obtaining weights.

## Expected layout

```
data/
  ptbxl/
    ptbxl_database.csv  scp_statements.csv  records500/...
  mimic-iv-ecg/
    record_list.csv  files/...
  mimiciv-hosp/
    admissions.csv.gz  patients.csv.gz
models/
  ecg-fm/  hubert-ecg/  ecgfounder/     # optional real backbones
embeddings/                              # cached .npy (gitignored)
```
