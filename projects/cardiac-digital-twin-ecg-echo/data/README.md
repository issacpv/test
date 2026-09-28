# Data acquisition

Expected layout (nothing here is committed):

```
data/
  ptb-xl/1.0.3/
    ptbxl_database.csv  scp_statements.csv  records100/  records500/
  ptb-xl-plus/1.0.1/
    features/12sl_features.csv  features/ecgdeli_features.csv ...
  echonet-dynamic/
    FileList.csv  VolumeTracings.csv  Videos/*.avi
  echonet-lvh/
    MeasurementsList.csv  Batch1..4/*.avi
  mimic-iv-ecg/1.0/
    record_list.csv  machine_measurements.csv  files/p1000/p10000032/s.../...
  mimic-iv-echo/0.1/
    echo-record-list.csv  files/p10/p10000032/s.../*.dcm
  mimiciv/3.1/hosp/  icu/          # linking + labels (admissions, diagnoses_icd, procedureevents)
  echonext/1.0.0/                  # paired ECG-echo benchmark (path: verify on PhysioNet)
  derived/
    pairs_mimic.csv  ecg_features_ptbxl.csv  twin_state_*.csv
```

## 1. PTB-XL and PTB-XL+ (open, no credentials)

```
python scripts/download_data.py --ptbxl --sample     # metadata + 5 records (~1 MB)
python scripts/download_data.py --ptbxl              # full 100 Hz + 500 Hz records (~3 GB)
# equivalent shell:
wget -r -N -c -np -nH --cut-dirs=1 -P data https://physionet.org/files/ptb-xl/1.0.3/
wget -r -N -c -np -nH --cut-dirs=1 -P data https://physionet.org/files/ptb-xl-plus/1.0.1/
```

## 2. EchoNet-Dynamic / EchoNet-LVH (free registration + Stanford AIMI research-use agreement)

1. Go to https://echonet.github.io/dynamic/ (and https://echonet.github.io/lvh/), follow "Access the data" to the Stanford AIMI Shared Datasets portal, register, and accept the research-use agreement.
2. You receive a download link (Azure/Box style). Extract to `data/echonet-dynamic/` so that `FileList.csv` and `Videos/` sit at that level.
3. `python scripts/download_data.py --check` verifies the layout and prints label summaries.

## 3. MIMIC-IV-ECG, MIMIC-IV-ECHO, MIMIC-IV, EchoNext (credentialed PhysioNet)

Requirements: PhysioNet account, CITI "Data or Specimens Only Research" training, signed DUA for each project. Then:

```
export PHYSIONET_USERNAME=...   # never commit these
export PHYSIONET_PASSWORD=...
python scripts/download_data.py --mimic-ecg --metadata-only   # record_list.csv + machine_measurements.csv
python scripts/download_data.py --mimic-echo --metadata-only  # echo-record-list.csv
# full waveform/DICOM mirrors (large; run on approved storage):
wget -r -N -c -np -nH --cut-dirs=1 -P data --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" https://physionet.org/files/mimic-iv-ecg/1.0/
wget -r -N -c -np -nH --cut-dirs=1 -P data --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" https://physionet.org/files/mimic-iv-echo/0.1/
```

MIMIC-IV-ECHO is multi-TB of DICOM; download only the studies that pair with an ECG (`derived/pairs_mimic.csv`, produced by `cardiac_twin.data.pair_ecg_echo`) using `--include-list`.

EchoNext: locate the project page on PhysioNet (search "EchoNext"), confirm the access tier and version, then mirror `https://physionet.org/files/<project>/<version>/` with the same wget pattern.

## 4. Derived tables

`derived/` is produced by the pipeline: ECG features per record, echo labels per study, pairs, twin states. These tables contain patient-level data from credentialed sources and must stay inside `data/` (git-ignored).
