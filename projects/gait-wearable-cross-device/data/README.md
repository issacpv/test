# Data acquisition

Expected layout (nothing committed):

```
data/
  physionet/
    gaitpdb/1.0.0/      # GaCo01_01.txt, GaPt03_01.txt, ..., demographics.txt
    gaitndd/1.0.0/      # control1.ts, park1.ts, hunt1.ts, als1.ts, ... (+ .let footswitch files)
    ltmm/1.0.0/         # WFDB records (3-day lower-back accelerometer)
    pads-parkinsons-disease-smartwatch/1.0.0/   # movement/*.json or csv per subject and task
  uci/daphnet/          # dataset/S01R01.txt ... (64 Hz, mg units)
  kaggle/tlvmc-fog/     # train/tdcsfog/*.csv, train/defog/*.csv, tdcsfog_metadata.csv, ...
  weargait_pd/          # per paper's data-availability statement
  mpower/               # Synapse tables + walking JSON files
  ppmi/                 # LONI IDA downloads: Verily Study Watch derived measures, MDS-UPDRS
  derived/
    registry.csv  windows_<dataset>.npz  features_<dataset>.parquet
```

## 1. Open PhysioNet datasets (no credentials)

```
python scripts/download_data.py --sample            # demographics + two gaitpdb records (~2 MB)
python scripts/download_data.py --all-open          # gaitpdb, gaitndd, ltmm, PADS, Daphnet
# equivalent wget for one project:
wget -r -N -c -np -nH --cut-dirs=1 -P data/physionet https://physionet.org/files/gaitpdb/1.0.0/
```

gaitpdb file naming: `<Study><Group><Subject>_<Walk>.txt`, e.g. `GaPt03_01.txt` (Ga = Galit Yogev study, Pt = patient, Co = control); columns: time, 8 left VGRF sensors, 8 right, total left, total right (100 Hz, N). `demographics.txt` holds group, age, sex, H&Y, UPDRS, gait speed.

## 2. Daphnet FoG (UCI, CC BY)

Downloaded by `--all-open` from the UCI static archive (`https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip`) and unzipped under `data/uci/daphnet/`. Files `S<subject>R<run>.txt`: time (ms), 3 x 3 accelerometer axes (ankle, thigh, trunk; mg), annotation (0 = not part of experiment, 1 = no freeze, 2 = freeze).

## 3. Kaggle TLVMC FoG contest data

Accept the competition rules, put your API token in `~/.kaggle/kaggle.json`, then:

```
python scripts/download_data.py --kaggle
```

(runs `kaggle competitions download -c tlvmc-parkinsons-freezing-gait-prediction`). Sub-folders `tdcsfog` (128 Hz, lab) and `defog` (100 Hz, home) contain lower-back accelerometer CSVs with FoG event labels; `daily` holds unlabelled week-long recordings.

## 4. mPower (Synapse; registration + qualified-researcher terms)

```
pip install synapseclient
synapse login -u <user> -p <password>      # or ~/.synapseConfig
python scripts/download_data.py --mpower   # prints the Synapse project to open and how to query the walking table
```

Data are under the mPower Public Researcher Portal (Synapse project `syn4993293`); walking-activity JSON files are attached to the table rows and fetched with `syn.downloadTableColumns`.

## 5. PPMI Verily Study Watch data (application + DUA; free)

1. Apply at https://www.ppmi-info.org/access-data-specimens/download-data (online application, PPMI Data Use Agreement).
2. After approval, log in to the LONI Image & Data Archive (https://ida.loni.usc.edu), open PPMI > Download > Study Data, and download the Verily Study Watch derived measures (physical activity, sleep, vital-sign features at hourly resolution) together with `MDS-UPDRS Part III`, `Participant Status` and `Demographics` tables. Raw Verily sensor data are made available to qualified researchers on separate request through PPMI/Verily.
3. Place CSVs under `data/ppmi/`. `scripts/download_data.py --ppmi` prints these steps and verifies the folder.

## 6. WearGait-PD

Follow the data-availability statement of the Scientific Data (2026) paper for the host and licence; place under `data/weargait_pd/`.
