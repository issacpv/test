# Data acquisition

Nothing in this folder is committed except this file. All commands run from the project root. Expected total: ~100 GB.

## Expected layout

```
data/
  tuev/                        # TUH EEG Events Corpus (registration)
    edf/train/<patient>/<file>.edf  (+ per-channel annotation files, .rec / .tse-style)
    edf/eval/...
  openneuro/
    ds003029/                  # Epilepsy-iEEG-Multicenter-Dataset (open; SOZ + outcome)
    ds003876/                  # Epilepsy-iEEG-Interictal-Multicenter-Dataset (open; sleep/awake)
    <sleep-IED accession>/     # two-centre annotated sleep iEEG (open, BIDS)
  omni_ieeg/                   # Omni-iEEG release (open; see arXiv 2602.16072 for the host)
  mni_atlas/                   # MNI Open iEEG Atlas wake + sleep segments (open, web download)
  bonn/                        # Bonn sets Z, O, N, F, S (open)
  manifests/                   # produced by scripts (safe to commit)
    records.csv  events.csv
```

## 1. TUEV (free registration)

1. Fill in the request form at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ (choose "TUH EEG Events Corpus"). Credentials arrive by e-mail.
2. Export them (never commit):

```bash
export TUH_USERNAME=...
export TUH_PASSWORD=...
export TUEV_VERSION=v2.0.1          # check the current version on the corpus page
python scripts/download_data.py --dataset tuev            # full corpus
python scripts/download_data.py --dataset tuev --sample   # eval split only
```

The script wraps `rsync -auxvL --partial <user>@www.isip.piconepress.com:data/tuh_eeg_events/$TUEV_VERSION/ data/tuev/`. Annotations are per channel with start/stop times and one of six labels (spsw, gped, pled, eyem, artf, bckg); the manifest builder keeps every label so that periodic discharges can be included or excluded downstream.

## 2. OpenNeuro iEEG datasets (open)

```bash
# development subset (metadata + first subject) using the public S3 listing API, pure python
python scripts/download_data.py --dataset openneuro --accession ds003029 --sample
python scripts/download_data.py --dataset openneuro --accession ds003876 --sample
# full downloads
openneuro-py download --dataset ds003029 --target-dir data/openneuro/ds003029
aws s3 sync --no-sign-request s3://openneuro.org/ds003029 data/openneuro/ds003029
```

For the two-centre annotated sleep iEEG dataset (*Sci. Data*, 2024), take the accession from the paper's Data Availability statement and pass it with `--accession`. SOZ channel flags in ds003029 are in the per-run `*_channels.tsv` (status/description columns) and outcome in `participants.tsv`; verify column names in the dataset's README before relying on them.

## 3. Omni-iEEG (open)

Follow the repository link in arXiv 2602.16072 (the dataset is a BIDS re-organization of OpenNeuro sources plus annotation tables). Place it under `data/omni_ieeg/`. If it is hosted on OpenNeuro, `--dataset openneuro --accession <id>` works unchanged.

## 4. MNI Open iEEG Atlas (open, web)

Go to https://mni-open-ieegatlas.research.mcgill.ca/, accept the terms, and download the wake atlas (Frauscher et al., 2018) and the sleep atlas (von Ellenrieder et al., 2020) data packages. Unpack under `data/mni_atlas/`. Each channel comes with its anatomical region label; keep the channel table, it is the "normal tissue" reference for false-positive calibration.

## 5. Bonn EEG (open)

Download the five zip archives (Z, O, N, F, S) from https://www.upf.edu/web/ntsa/downloads and unpack into `data/bonn/<set>/`. Each set is 100 text files of 4,097 samples at 173.61 Hz. `python scripts/download_data.py --dataset bonn` prints these instructions and verifies the layout if the files are present.

## 6. Manifests

```bash
python scripts/download_data.py --build-manifests
```

Writes `records.csv` (dataset, record_id, subject_id, path, modality, fs, is_normal, vigilance_available) and `events.csv` (dataset, record_id, channel, onset_s, offset_s, label, annotator). Only public ids and times are stored.
