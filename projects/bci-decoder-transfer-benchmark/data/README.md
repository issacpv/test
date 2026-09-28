# Data acquisition

Everything is open. Two routes: MOABB (recommended, handles all datasets) and direct
PhysioNet download for EEGMMIDB (no registration). Never commit raw EEG.

## 1. Install

```bash
pip install "moabb>=1.1" "mne>=1.6" "pyriemann>=0.6"
export MNE_DATA=$PWD/data/mne_data          # where MOABB/MNE caches downloads
python -c "import mne; mne.set_config('MNE_DATA', '$PWD/data/mne_data')"
```

## 2. List the motor-imagery datasets available in your MOABB version

```bash
python scripts/download_data.py --list
```

This prints, for each `moabb.datasets` MI class: subjects, sessions, channel count, classes,
sampling rate and the citation MOABB attaches, and writes `data/datasets.json`.

## 3. Download

Smoke test (subject 1 of three small datasets):

```bash
python scripts/download_data.py --sample
```

Selected datasets, all subjects (resumable; MOABB skips files already cached):

```bash
python scripts/download_data.py --moabb BNCI2014_001 BNCI2014_004 PhysionetMI Cho2017 Lee2019_MI
```

Everything MI:

```bash
python scripts/download_data.py --moabb all
```

Some hosts are slow or occasionally offline (GigaDB for Cho2017 / Lee2019; Zenodo for
Dreyer2023; Figshare for Stieger2021). Re-run the command; MOABB resumes.

## 4. PhysioNet EEGMMIDB without MOABB (open, wget)

```bash
# whole database (~3.4 GB, 109 subjects x 14 runs, EDF+)
wget -r -N -c -np -nH --cut-dirs=2 -P data/eegmmidb https://physionet.org/files/eegmmidb/1.0.0/
# or a few runs of one subject (runs 4, 8, 12 = imagined left/right fist; 6, 10, 14 = fists/feet)
python scripts/download_data.py --physionet --subjects 1 2 --sample
```

No credentials are required for EEGMMIDB (it is an open PhysioNet database).

## 5. Foundation-model weights (optional deep arm)

LaBraM and CBraMod publish checkpoints on their GitHub repositories; download them into
`data/weights/` and record the commit hash. Check their pre-training dataset lists for
overlap with your target datasets (see README, leakage prevention).

## Expected layout

```
data/
  README.md
  datasets.json              (output of --list)
  mne_data/                  (MOABB / MNE cache; dataset-specific subfolders)
    MNE-bnci-data/
    MNE-eegbci-data/
    ...
  eegmmidb/                  (only if downloaded directly from PhysioNet)
    S001/S001R04.edf ...
  epochs/                    (derived: harmonised epochs per dataset/subject/session, .npz)
    BNCI2014_001/sub-01_ses-0.npz
  covs/                      (derived: shrinkage covariances per trial)
  weights/                   (optional foundation-model checkpoints)
```

`.npz` epoch files store `X (n_trials, n_channels, n_times)`, `y`, `ch_names`, `sfreq`, and the
preprocessing config hash used by the benchmark.
