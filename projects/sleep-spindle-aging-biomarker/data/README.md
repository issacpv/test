# Data acquisition

Nothing in this folder is committed except this file. Open datasets can be
fetched by the script; NSRR datasets require a free account, a signed data-use
agreement per dataset, and the `NSRR_TOKEN` environment variable.

## Expected layout

```
data/
  sleep-edf/                       # PhysioNet sleep-edfx 1.0.0
    sleep-cassette/SC4001E0-PSG.edf  SC4001EC-Hypnogram.edf ...
    sleep-telemetry/ST7011J0-PSG.edf ...
    SC-subjects.xls  ST-subjects.xls
  hmc/                             # PhysioNet hmc-sleep-staging
    recordings/SN001.edf  SN001_sleepscoring.edf ...
  nsrr/
    shhs/polysomnography/edfs/shhs1/shhs1-200001.edf
    shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml
    shhs/datasets/shhs1-dataset-0.21.0.csv   shhs/datasets/shhs-cvd-events-dataset-*.csv
    mesa/polysomnography/edfs/mesa-sleep-0001.edf
    mesa/polysomnography/annotations-events-nsrr/mesa-sleep-0001-nsrr.xml
    mesa/datasets/mesa-sleep-dataset-*.csv
    mros/polysomnography/edfs/visit1/mros-visit1-aa0001.edf  (+ annotations-events-nsrr/visit1/*.xml)
    cfs/polysomnography/edfs/cfs-visit5-800002.edf  (+ annotations-events-nsrr/*.xml)
  dod/  dod-h/*.h5  dod-o/*.h5    # Dreem Open Datasets (optional)
  openneuro/<dsID>/                # BIDS hd-EEG sleep datasets (optional)
  events/<cohort>/<night>.parquet  # spindle / SO tables produced by the pipeline
  metrics/<cohort>.csv             # per-night coupling metrics
```

## 1. Sleep-EDF Expanded (open, PhysioNet)

```bash
python scripts/download_data.py --dataset sleep-edf --sample    # SC4001 night + subject tables
python scripts/download_data.py --dataset sleep-edf             # all 197 PSGs (~8 GB)
# equivalent:
wget -r -N -c -np -nH --cut-dirs=3 -P data/sleep-edf https://physionet.org/files/sleep-edfx/1.0.0/
```

Hypnograms are EDF+ annotation files (`*-Hypnogram.edf`) read with `mne.read_annotations`. Ages/sex are in `SC-subjects.xls` / `ST-subjects.xls`.

## 2. HMC Sleep Staging (open, PhysioNet)

```bash
python scripts/download_data.py --dataset hmc --sample
```

## 3. NSRR cohorts: SHHS, MESA, MrOS, CFS (free registration + DUA)

1. Create an account at https://sleepdata.org, request access to each dataset (SHHS, MESA, MrOS, CFS) and sign the DUA; approval typically takes days.
2. Get your token from https://sleepdata.org/token and export it:

```bash
export NSRR_TOKEN=...            # never commit
gem install nsrr                 # Ruby CLI used by NSRR (https://github.com/nsrr/nsrr-gem)
python scripts/download_data.py --dataset shhs --sample      # 2 EDFs + XMLs + dataset CSVs
python scripts/download_data.py --dataset mesa --sample
python scripts/download_data.py --dataset mros --sample
python scripts/download_data.py --dataset cfs --sample
python scripts/download_data.py --dataset shhs               # full (~250 GB)
```

The script wraps, e.g.:

```bash
nsrr download shhs/polysomnography/edfs/shhs1 --file="shhs1-20000*" --token=$NSRR_TOKEN
nsrr download shhs/polysomnography/annotations-events-nsrr/shhs1 --token=$NSRR_TOKEN
nsrr download shhs/datasets --token=$NSRR_TOKEN
```

Useful files: `*-dataset-*.csv` (covariates incl. harmonized `nsrr_*` variables), `shhs-cvd-events-dataset-*.csv` (adjudicated outcomes), `mesa-sleep-dataset-*.csv` (includes cognition variables from MESA exams where released), `mros-visit1-dataset-*.csv` (3MS, Trails B).

## 4. Dreem Open Datasets (open, public S3; optional)

```bash
pip install awscli
aws s3 sync --no-sign-request s3://dreem-dod-h data/dod/dod-h
aws s3 sync --no-sign-request s3://dreem-dod-o data/dod/dod-o
```

(Guillot et al., 2020, IEEE TNSRE. Check the bucket region/name in the paper's repository if the sync fails.)

## 5. OpenNeuro hd-EEG sleep datasets (open; optional, for H6)

Search https://openneuro.org for modality EEG + "sleep" (overnight, BIDS with hypnogram TSVs), then:

```bash
pip install openneuro-py
python scripts/download_data.py --dataset openneuro --openneuro-id dsXXXXXX --sample   # first subject only
# or: openneuro-py download --dataset dsXXXXXX --include sub-01
# or: datalad install https://github.com/OpenNeuroDatasets/dsXXXXXX.git
```

## 6. Per-night pipeline outputs

```bash
python -m spindle_age.run_night data/sleep-edf/sleep-cassette/SC4001E0-PSG.edf --cohort sleep-edf   # (to be added)
```

writes `data/events/sleep-edf/SC4001E0.parquet` and appends to `data/metrics/sleep-edf.csv`.

## Hygiene

- `NSRR_TOKEN` only in the environment or a git-ignored `.env`.
- For the full NSRR cohorts use a stream-and-delete loop (download night, detect events, store parquet, delete EDF) to keep disk under 50 GB.
