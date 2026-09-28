# Data acquisition

Nothing in this folder is committed except this file. All commands run from the project root. Expected total: ~150 GB.

## Expected layout

```
data/
  tusz/                        # TUH EEG Seizure Corpus v2.0.3 (registration)
    edf/train/<patient>/<session>/<montage>/<file>.edf  (+ .csv_bi)
    edf/dev/...  edf/eval/...
  tuab/                        # TUH EEG Abnormal Corpus (registration; age-labelled background EEG)
    edf/train/normal/...  edf/train/abnormal/...  edf/eval/...
  chbmit/                      # PhysioNet chbmit 1.0.0 (open)
    chb01/chb01_01.edf ... chb01-summary.txt   SUBJECT-INFO
  siena/                       # PhysioNet siena-scalp-eeg 1.0.0 (open)
    PN00/PN00-1.edf ... Seizures-list-PN00.txt   subject_info.csv
  helsinki/                    # Zenodo neonatal corpus (open)
    eeg1.edf ... eeg79.edf   annotations_2017_A.csv ...   clinical_information.csv
  manifests/                   # produced by scripts (safe to commit)
    records.csv  events.csv  ages.csv
```

## 1. TUH EEG Seizure Corpus (TUSZ) and TUH Abnormal Corpus (TUAB) - free registration

1. Fill in the request form at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ (tick the corpora you need). Credentials arrive by e-mail.
2. Export them (never commit):

```bash
export TUH_USERNAME=...
export TUH_PASSWORD=...
python scripts/download_data.py --dataset tusz            # ~70 GB
python scripts/download_data.py --dataset tusz --sample   # dev split only
python scripts/download_data.py --dataset tuab            # TUAB (set TUAB_VERSION if the server path differs)
```

The script wraps `rsync -auxvL --partial <user>@www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.3/ data/tusz/` (and `data/tuh_eeg_abnormal/$TUAB_VERSION/` for TUAB; check the current version on the corpus page). Patient age is embedded in the EDF local-patient-identification field as `Age:NN`; `--build-manifests` parses it with `eegage.ages.parse_edf_header_age` and writes `ages.csv`.

## 2. CHB-MIT (open, PhysioNet)

```bash
python scripts/download_data.py --dataset chbmit            # ~42 GB
python scripts/download_data.py --dataset chbmit --sample   # SUBJECT-INFO + chb01 summary + two records
# equivalent:
wget -r -N -c -np -nH --cut-dirs=3 -P data/chbmit https://physionet.org/files/chbmit/1.0.0/
```

Ages and sex are in `SUBJECT-INFO` (tab-separated: case, gender, age).

## 3. Siena Scalp EEG (open, PhysioNet)

```bash
python scripts/download_data.py --dataset siena
```

Ages are in `subject_info.csv`.

## 4. Helsinki neonatal EEG (open, Zenodo)

Stevenson et al. (2019), "A dataset of neonatal EEG recordings with seizure annotations". Find the Zenodo record id (search the title; the concept record resolves to the latest version), then:

```bash
export HELSINKI_ZENODO_RECORD=<numeric record id>
python scripts/download_data.py --dataset helsinki --sample   # first 2 EDFs + annotations + clinical info
python scripts/download_data.py --dataset helsinki            # all 79 EDFs
```

The script lists files through the Zenodo REST API (`https://zenodo.org/api/records/<id>`) and downloads each with resume support.

## 5. Manifests

```bash
python scripts/download_data.py --build-manifests
```

Writes `records.csv` (dataset, record_id, subject_id, path, fs, age_years, age_bin), `events.csv` (dataset, record_id, onset_s, offset_s) and `ages.csv` (dataset, subject_id, age_years, age_bin, n_records). Only public ids, times and binned ages are stored.
