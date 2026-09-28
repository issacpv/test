# Data acquisition

Nothing under `data/` is committed. All commands below are run from the project root.
Expected final layout:

```
data/
  ptbxl/                      # PhysioNet ptb-xl 1.0.3 (open)
    ptbxl_database.csv
    scp_statements.csv
    records500/00000/00001_hr.{hea,dat} ...
  ecg-arrhythmia/             # PhysioNet "A large scale 12-lead ECG database for arrhythmia study" 1.0.0
    WFDBRecords/01/010/JS00001.{hea,mat} ...   (Chapman-Shaoxing + Ningbo, 45,152 records)
    ConditionNames_SNOMED-CT.csv
  challenge2021/              # PhysioNet Challenge 2021 1.0.3 training set (open)
    training/georgia/g1/E00001.{hea,mat} ...
    training/cpsc_2018/g1/A0001.{hea,mat} ...
    training/cpsc_2018_extra/...
    training/chapman_shaoxing/... , training/ningbo/...   (duplicates of ecg-arrhythmia, Challenge header format)
    dx_mapping_scored.csv, dx_mapping_unscored.csv
  code15/                     # Zenodo record 4916206 (open, CC-BY 4.0)
    exams.csv
    exams_part0.hdf5 ... exams_part17.hdf5
  mimic-iv-ecg/               # PhysioNet mimic-iv-ecg 1.0 (credentialed)
    record_list.csv, machine_measurements.csv, waveform_note_links.csv
    files/p1000/p10000032/s40689238/40689238.{hea,dat} ...
  mimiciv/3.1/hosp/           # PhysioNet mimiciv 3.1 hosp module (credentialed)
    patients.csv.gz, admissions.csv.gz, labevents.csv.gz, d_labitems.csv.gz
  mimic-iv-ecg-ext-icd/       # optional ICD labels (credentialed)
```

## 1. Open datasets (no account needed)

### PTB-XL 1.0.3

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/ptbxl \
  https://physionet.org/files/ptb-xl/1.0.3/
# or only the 500 Hz records + metadata (about 1.8 GB):
wget -N -P data/ptbxl https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv \
                      https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
wget -r -N -c -np -nH --cut-dirs=3 -P data/ptbxl https://physionet.org/files/ptb-xl/1.0.3/records500/
```

Citation: Wagner P. et al., Sci Data 2020. Use `strat_fold` 1-8 train, 9 val, 10 test.

### Chapman-Shaoxing + Ningbo ("ecg-arrhythmia" 1.0.0, ~6 GB)

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/ecg-arrhythmia \
  https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

Records JS00001-JS10646 are Chapman-Shaoxing (Zheng et al., Sci Data 2020); JS10647 onwards are Ningbo (Zheng et al., Sci Rep 2020). Labels are SNOMED-CT codes in the `.hea` `#Dx:` comment line.

### PhysioNet Challenge 2021 training data (Georgia, CPSC-2018, CPSC-Extra; ~4 GB for those three)

```bash
BASE=https://physionet.org/files/challenge-2021/1.0.3
wget -N -P data/challenge2021 $BASE/dx_mapping_scored.csv $BASE/dx_mapping_unscored.csv
for src in georgia cpsc_2018 cpsc_2018_extra; do
  wget -r -N -c -np -nH --cut-dirs=3 -P data/challenge2021 $BASE/training/$src/
done
```

Alternatively use the Challenge tarballs listed on https://physionet.org/content/challenge-2021/1.0.3/ (`WFDB_Ga.tar.gz`, `WFDB_CPSC2018.tar.gz`, `WFDB_CPSC2018_2.tar.gz`, `WFDB_ChapmanShaoxing.tar.gz`, `WFDB_Ningbo.tar.gz`). Header comments contain `#Age`, `#Sex`, `#Dx` (comma-separated SNOMED codes).

### CODE-15% (Zenodo, ~40 GB)

```bash
python scripts/download_data.py --dataset code15 --out data/code15          # all 18 HDF5 parts + exams.csv
python scripts/download_data.py --dataset code15 --out data/code15 --sample # exams.csv + part 0 only
```

The script uses the Zenodo REST API (`https://zenodo.org/api/records/4916206`) to list files and verifies MD5.
`exams.csv` columns: `exam_id, age, is_male, nn_predicted_age, 1dAVb, RBBB, LBBB, SB, ST, AF, patient_id, death, timey, normal_ecg, trace_file`. Traces: `tracings` dataset of shape (N, 4096, 12) at 400 Hz, zero-padded, lead order DI, DII, DIII, AVR, AVL, AVF, V1-V6. Citation: Ribeiro A.H. et al., Nat Commun 2020; Zenodo 2021.

## 2. Credentialed datasets (PhysioNet account + CITI training + signed DUA)

1. Create a PhysioNet account, complete the CITI "Data or Specimens Only Research" course, upload the certificate, and sign the DUA for **MIMIC-IV-ECG**, **MIMIC-IV** and (optionally) **MIMIC-IV-ECG-Ext-ICD**.
2. Export credentials as environment variables (never write them into files that are committed):

```bash
export PHYSIONET_USERNAME=your_user
export PHYSIONET_PASSWORD='your_password'
```

3. Download:

```bash
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg           # metadata only (~1 GB)
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --waveforms  # + all WFDB files (~90 GB)
python scripts/download_data.py --dataset mimiciv-hosp --out data/mimiciv/3.1/hosp        # patients, admissions, labevents, d_labitems
```

which is equivalent to

```bash
wget -r -N -c -np -nH --cut-dirs=1 --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" \
  -P data https://physionet.org/files/mimic-iv-ecg/1.0/
```

`record_list.csv` has `subject_id, study_id, file_name, ecg_time, path`; `machine_measurements.csv` has interval measurements and the 18 machine-report text fields `report_0..report_17`. Troponin T is `labevents.itemid = 51003` (`d_labitems`), mortality date is `patients.dod`.

## 3. Verify

```bash
python scripts/download_data.py --verify --out data
```

prints counts of records found per source and flags missing metadata files.
