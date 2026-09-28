# Data acquisition

Nothing in this folder is committed except this file. All commands run from the project root. Expected total: ~3 TB if all NSRR cohorts are fetched; ~8 GB for Sleep-EDF alone.

## Expected layout

```
data/
  sleep-edfx/                    # PhysioNet sleep-edfx 1.0.0 (open)
    sleep-cassette/SC4001E0-PSG.edf  SC4001EC-Hypnogram.edf ...
    sleep-telemetry/ST7011J0-PSG.edf ST7011JP-Hypnogram.edf ...
    SC-subjects.xls  ST-subjects.xls
  nsrr/
    shhs/
      polysomnography/edfs/shhs1/shhs1-200001.edf ...
      polysomnography/edfs/shhs2/...
      polysomnography/annotations-events-profusion/shhs1/shhs1-200001-profusion.xml ...
      datasets/shhs1-dataset-<ver>.csv  shhs2-dataset-<ver>.csv
      datasets/shhs-cvd-summary-dataset-<ver>.csv
      datasets/shhs-harmonized-dataset-<ver>.csv
    mesa/
      polysomnography/edfs/mesa-sleep-0001.edf ...
      polysomnography/annotations-events-profusion/mesa-sleep-0001-profusion.xml ...
      datasets/mesa-sleep-dataset-<ver>.csv  mesa-sleep-harmonized-dataset-<ver>.csv
    mros/  cfs/                   # same structure
  manifests/                      # produced by scripts; safe to commit (no per-subject PHI)
    recordings.csv
```

## 1. Sleep-EDF Expanded (open, PhysioNet)

No account required.

```bash
python scripts/download_data.py --dataset sleep-edfx            # all 197 recordings (~8 GB)
python scripts/download_data.py --dataset sleep-edfx --sample   # SC4001 + SC4002 PSG/hypnogram pairs
# equivalent to
wget -r -N -c -np -nH --cut-dirs=3 -P data/sleep-edfx https://physionet.org/files/sleep-edfx/1.0.0/
```

Hypnograms are EDF+ annotation files (`*-Hypnogram.edf`) with labels "Sleep stage W/1/2/3/4/R", "Movement time" and "?". Read them with `mne.read_annotations` or `pyedflib.EdfReader.readAnnotations`, then convert to 30-s epochs with `stagebias.hypnogram.annotations_to_epochs` (stages 3 and 4 are merged into N3).

## 2. NSRR cohorts: SHHS, MESA, MrOS, CFS (DUA, free)

1. Create an account at https://sleepdata.org and submit a Data Access Request for each dataset (SHHS, MESA, MrOS, CFS). You must upload a signed Data Access and Use Agreement; institutional sign-off is usually required. Approval typically takes days to a few weeks.
2. Once approved, find your download token at https://sleepdata.org/token and export it (never commit it):

```bash
export NSRR_TOKEN=...
```

3. Install the official NSRR command-line client (Ruby gem) and download:

```bash
gem install nsrr
# stage annotations only (small, start here)
nsrr download shhs/polysomnography/annotations-events-profusion --token=$NSRR_TOKEN
nsrr download shhs/datasets --token=$NSRR_TOKEN
# EDFs (large)
nsrr download shhs/polysomnography/edfs/shhs1 --token=$NSRR_TOKEN
nsrr download mesa/polysomnography/annotations-events-profusion --token=$NSRR_TOKEN
nsrr download mesa/datasets --token=$NSRR_TOKEN
```

`scripts/download_data.py --dataset shhs|mesa|mros|cfs` wraps these `nsrr` calls (it shells out to the gem, runs a dry-run first, and refuses to run without `NSRR_TOKEN`). Run the client from `data/nsrr/` so the paths match the layout above.

Key files:

- `polysomnography/annotations-events-profusion/*.xml`: Compumedics profusion XML with `<SleepStages>` (30-s codes: 0 W, 1 N1, 2 N2, 3 N3, 4 N4, 5 REM, 9 unscored) and `<ScoredEvents>` (obstructive/central apnoea, hypopnoea, arousal, desaturation, with `Start` and `Duration` in seconds). Parsed by `stagebias.hypnogram.parse_profusion_xml`.
- `datasets/shhs1-dataset-*.csv`: per-subject covariates including AHI (e.g., `ahi_a0h3a`, `ahi_a0h4`), arousal index, age, sex, BMI, blood pressure, medication flags; `datasets/shhs-cvd-summary-dataset-*.csv`: adjudicated incident CHD/CVD/CHF/stroke, vital status and censoring dates; `datasets/*-harmonized-dataset-*.csv`: NSRR harmonized variables (prefix `nsrr_`, e.g., harmonized AHI definitions) - prefer these for cross-cohort work.
- MESA: `mesa-sleep-dataset-*.csv` (AHI, arousal index, Exam 5 covariates). CVD events beyond what NSRR distributes require a MESA ancillary-study/data request through the MESA Coordinating Center.

## 3. Manifests

```bash
python scripts/download_data.py --build-manifest
```

writes `data/manifests/recordings.csv` with one row per recording (cohort, id, EDF path, XML path, epoch count, N3 minutes from XML) and flags missing files. The manifest contains no clinical variables and may be committed.
