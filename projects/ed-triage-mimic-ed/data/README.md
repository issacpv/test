# Data acquisition (all MIMIC modules are PhysioNet-credentialed)

Nothing under `data/` is committed. Expected layout after download:

```
data/
  mimic-iv-ed/2.2/ed/
    edstays.csv.gz  triage.csv.gz  vitalsign.csv.gz  medrecon.csv.gz  pyxis.csv.gz  diagnosis.csv.gz
  mimiciv/3.1/
    hosp/patients.csv.gz  hosp/admissions.csv.gz  hosp/transfers.csv.gz
    hosp/procedures_icd.csv.gz  hosp/labevents.csv.gz (optional, 13 GB)  hosp/d_labitems.csv.gz
    icu/icustays.csv.gz
  mimic-iv-note/2.2/note/
    discharge.csv.gz  radiology.csv.gz            (optional; outcome adjudication only)
  mimic-iv-ed-demo/2.2/ed/                        (open demo, 100 patients; used by --sample)
    edstays.csv.gz  triage.csv.gz ...
  mimic-iv-demo/2.2/hosp/, icu/                   (open demo of MIMIC-IV to pair with the ED demo)
```

## 1. Credentialing (once)

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training and upload the completion report
   (https://physionet.org/about/citi-course/).
3. Apply for credentialed access and sign the DUA for each project:
   MIMIC-IV-ED (https://physionet.org/content/mimic-iv-ed/2.2/), MIMIC-IV (https://physionet.org/content/mimiciv/3.1/),
   MIMIC-IV-Note (https://physionet.org/content/mimic-iv-note/2.2/).

## 2. Credentials via environment variables

```bash
export PHYSIONET_USERNAME=your_user
export PHYSIONET_PASSWORD='your_password'      # never commit; consider a .env excluded by .gitignore
```

## 3. Download

```bash
python scripts/download_data.py --sample --out data        # open demo (no credentials)
python scripts/download_data.py --module ed --out data      # MIMIC-IV-ED 2.2 (~ 200 MB compressed)
python scripts/download_data.py --module hosp --out data    # patients, admissions, transfers, procedures_icd, d_labitems
python scripts/download_data.py --module hosp --labevents --out data   # + labevents (13 GB)
python scripts/download_data.py --module icu --out data     # icustays
python scripts/download_data.py --module note --out data    # discharge + radiology (3.5 GB)
python scripts/download_data.py --verify --out data
```

Equivalent manual command (resumable):

```bash
wget -r -N -c -np -nH --cut-dirs=1 --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" \
  -P data https://physionet.org/files/mimic-iv-ed/2.2/
```

## 4. Key columns

- `ed/edstays`: `subject_id, hadm_id, stay_id, intime, outtime, gender, race, arrival_transport, disposition`
- `ed/triage`: `subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint`
- `hosp/patients`: `subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod`
- `hosp/admissions`: `subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, insurance, language, marital_status, race, hospital_expire_flag`
- `hosp/transfers`: `subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime`
- `icu/icustays`: `subject_id, hadm_id, stay_id, first_careunit, intime, outtime, los`

Age at ED arrival = `anchor_age + (year(intime) - anchor_year)`. Only `anchor_year_group` (e.g. "2017 - 2019") is real-calendar anchored; all other dates are shifted per patient.

## 5. Reference pipeline

Xie et al. 2022 (Sci Data) code: https://github.com/nliulab/mimic4ed-benchmark. Our `ed_triage.cohort` SQL reproduces their inclusion criteria and three outcomes; differences are documented in the module docstring.
