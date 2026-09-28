# Data acquisition

Nothing in this folder is committed. Expected layout after the steps below:

```
data/
  mimic3wdb-matched/            # open; record-by-record via wfdb (only selected records)
    RECORDS, RECORDS-waveforms, RECORDS-numerics
    p0X/pXXXXXX/pXXXXXX-YYYY-MM-DD-hh-mm.{hea,dat} ...
  mimiciii/                     # credentialed; only the tables listed below
    DIAGNOSES_ICD.csv.gz  PATIENTS.csv.gz  ADMISSIONS.csv.gz  ICUSTAYS.csv.gz  INPUTEVENTS_MV.csv.gz
  mimic4wdb/                    # credentialed; MIMIC-IV Waveform v0.1.0
  mimiciv/hosp/                 # credentialed; diagnoses_icd, patients, admissions, transfers
  vitaldb/
    cases.csv  trks.csv  <caseid>.vital or .csv per case
  pulsedb/                      # PulseDB .mat files (large); optional
  derived/                      # beat tables written by the pipeline
```

## 1. MIMIC-III Waveform Database Matched Subset (open)

No credentialing. Get the record index and read records on demand with `wfdb`:

```bash
python scripts/download_data.py --mimic3-index            # RECORDS files only
python scripts/download_data.py --sample                  # one small matched record via wfdb (pn_dir)
```

In code:
```python
import wfdb
rec = wfdb.rdrecord("p000020-2183-04-28-17-47", pn_dir="mimic3wdb-matched/1.0/p00/p000020", sampto=125*60)
```
Only download records that pass the cohort filter (see `cohort.py`); the full subset is > 2 TB.

## 2. MIMIC-III Clinical v1.4 (credentialed)

1. PhysioNet account, CITI "Data or Specimens Only Research" training, sign the MIMIC-III DUA.
2. `export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...`
3. `python scripts/download_data.py --mimic3-clinical` runs
   `wget -N -c --user $PHYSIONET_USERNAME --password $PHYSIONET_PASSWORD https://physionet.org/files/mimiciii/1.4/<TABLE>.csv.gz`
   for DIAGNOSES_ICD, PATIENTS, ADMISSIONS, ICUSTAYS, INPUTEVENTS_MV, D_ITEMS.

## 3. MIMIC-IV v3.1 hosp module and MIMIC-IV Waveform v0.1.0 (credentialed)

Same credentials. `python scripts/download_data.py --mimic4` fetches `hosp/diagnoses_icd`, `hosp/patients`, `hosp/admissions`, `hosp/transfers` and the waveform `RECORDS` index. Waveform records are then read on demand with `wfdb.rdrecord(..., pn_dir="mimic4wdb/0.1.0/waves/...")`.

## 4. VitalDB (open API)

```bash
pip install vitaldb
python scripts/download_data.py --vitaldb-index          # cases.csv and trks.csv from https://api.vitaldb.net
python scripts/download_data.py --vitaldb-cases 10       # first 10 eligible cases (women 18-45 with ECG_II, PLETH, ART)
```
The script reports how many cases have `opname` matching caesarean/C-section so the "VitalDB obstetric cases (if any)" question is settled empirically. Accept the VitalDB data-use terms on https://vitaldb.net/dataset/ before use and cite Lee et al., 2022 Sci Data.

## 5. PulseDB (open, optional)

Follow https://github.com/pulselabteam/PulseDB (MATLAB `.mat` files hosted via the release links). Only the subject-id fields are needed for overlap exclusion; the segments are needed only if you retrain general-population models. Load with `scipy.io.loadmat` or `h5py` (v7.3 files).

## Checks

`python scripts/download_data.py --check` prints which parts are present and the number of records/cases found.
