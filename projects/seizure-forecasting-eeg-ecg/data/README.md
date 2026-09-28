# Data acquisition

Nothing in this folder is committed except this file. All commands run from the project root. Expected total: ~250 GB (SeizeIT2 is the bulk).

## Expected layout

```
data/
  seizeit2/                    # OpenNeuro ds005873 (open, BIDS)
    dataset_description.json  participants.tsv
    sub-001/ses-01/eeg/sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
    sub-001/ses-01/eeg/sub-001_ses-01_task-szMonitoring_run-01_events.tsv
    ...                        # (exact task/run labels come from the dataset; do not hard-code)
  siena/                       # PhysioNet siena-scalp-eeg 1.0.0 (open)
    PN00/PN00-1.edf ... Seizures-list-PN00.txt
    subject_info.csv
  szdb/                        # PhysioNet szdb 1.0.0 (open, ECG only)
    sz01.hea sz01.dat sz01.ari ... times.seize
  tusz/                        # TUH EEG Seizure Corpus v2.0.3 (registration)
    edf/train/<patient>/<session>/<montage>/<file>.edf  (+ .csv_bi)
  manifests/                   # produced by scripts (safe to commit)
    records.csv  events.csv
```

## 1. SeizeIT2 (open, OpenNeuro ds005873)

Three equivalent routes; pick one.

```bash
# (a) pure-python subset for development: metadata + first subject (uses the public S3 listing API)
python scripts/download_data.py --dataset seizeit2 --sample

# (b) full dataset with openneuro-py (pip install openneuro-py)
openneuro-py download --dataset ds005873 --target-dir data/seizeit2

# (c) full dataset with the AWS CLI from the public OpenNeuro bucket (no credentials needed)
aws s3 sync --no-sign-request s3://openneuro.org/ds005873 data/seizeit2
```

DataLad also works: `datalad install https://github.com/OpenNeuroDatasets/ds005873.git` then `datalad get`.

Seizure events are in the BIDS `*_events.tsv` files (columns `onset`, `duration`, `eventType`, ...); the modality of each channel (EEG / ECG / EMG / ACC) is described in `*_channels.tsv`. Record the dataset version you used in `manifests/records.csv` (OpenNeuro snapshots are immutable).

## 2. Siena Scalp EEG (open, PhysioNet)

```bash
python scripts/download_data.py --dataset siena            # everything (~20 GB)
python scripts/download_data.py --dataset siena --sample   # subject_info + PN00 first record
# equivalent:
wget -r -N -c -np -nH --cut-dirs=3 -P data/siena https://physionet.org/files/siena-scalp-eeg/1.0.0/
```

The EKG channel is stored alongside the EEG channels in each EDF (channel label contains `EKG`). Events are wall-clock times in `Seizures-list-PNxx.txt`; the manifest builder converts them to seconds from record start using the EDF start time.

## 3. Post-ictal heart rate oscillations (szdb; open, PhysioNet)

```bash
python scripts/download_data.py --dataset szdb
# equivalent:
wget -r -N -c -np -nH --cut-dirs=3 -P data/szdb https://physionet.org/files/szdb/1.0.0/
```

WFDB format (`.hea/.dat/.ari`); read with the `wfdb` Python package (`pip install wfdb`). Seizure times are in `times.seize`.

## 4. TUH EEG Seizure Corpus v2.0.3 (free registration)

1. Fill in the request form at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ (choose "TUH EEG Seizure Corpus"). Credentials arrive by e-mail.
2. Export them (never commit):

```bash
export TUH_USERNAME=...
export TUH_PASSWORD=...
python scripts/download_data.py --dataset tusz            # full corpus (~70 GB)
python scripts/download_data.py --dataset tusz --sample   # dev split only
```

The script wraps `rsync -auxvL --partial <user>@www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.3/ data/tusz/` (password via `sshpass` if installed, otherwise rsync prompts). Only records whose EDF header lists an EKG channel are used for the ECG arm; `--build-manifests` records this per file in `records.csv` (`has_ecg`, `ecg_channel`).

## 5. Manifests

```bash
python scripts/download_data.py --build-manifests
```

Writes `data/manifests/records.csv` (dataset, record_id, subject_id, path, fs, has_ecg, ecg_channel, start_clock, duration_s) and `data/manifests/events.csv` (dataset, record_id, onset_s, offset_s, event_type). These contain only ids and times that are already public and may be committed.
