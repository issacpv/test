# Data acquisition

Nothing in this folder is committed except this file. All commands are run from the project root. Expected total: ~150 GB.

## Expected layout

```
data/
  chbmit/                      # PhysioNet chbmit 1.0.0 (open)
    chb01/chb01_01.edf ... chb01-summary.txt
    ...
  siena/                       # PhysioNet siena-scalp-eeg 1.0.0 (open)
    PN00/PN00-1.edf ... Seizures-list-PN00.txt
    subject_info.csv
  tusz/                        # TUH EEG Seizure Corpus v2.0.3 (registration)
    edf/train/<patient>/<session>/<montage>/<file>.edf  (+ .csv_bi, .csv)
    edf/dev/...  edf/eval/...
  helsinki/                    # Zenodo neonatal corpus (open)
    eeg1.edf ... eeg79.edf
    annotations_2017_A.csv annotations_2017_B.csv annotations_2017_C.csv
    clinical_information.csv
  manifests/                   # produced by scripts (safe to commit)
    records.csv  events.csv
```

## 1. CHB-MIT (open, PhysioNet)

No account required. Either use the script:

```bash
python scripts/download_data.py --dataset chbmit            # everything (~42 GB)
python scripts/download_data.py --dataset chbmit --sample   # chb01 summary + first two records
```

or wget directly (this is what the script wraps):

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/chbmit https://physionet.org/files/chbmit/1.0.0/
```

Seizure onsets/offsets are in `chbXX-summary.txt` (seconds from file start). The `.edf.seizures` binary files are redundant.

## 2. Siena Scalp EEG (open, PhysioNet)

```bash
python scripts/download_data.py --dataset siena
# equivalent:
wget -r -N -c -np -nH --cut-dirs=3 -P data/siena https://physionet.org/files/siena-scalp-eeg/1.0.0/
```

Events are listed in `Seizures-list-PNxx.txt` as wall-clock times; the script's manifest builder converts them to seconds from record start using the EDF start time.

## 3. TUH EEG Seizure Corpus v2.0.3 (free registration)

1. Fill in the request form at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ (choose "TUH EEG Seizure Corpus"). You receive a username/password by e-mail.
2. Export credentials (never commit them):

```bash
export TUH_USERNAME=...   # as e-mailed
export TUH_PASSWORD=...
python scripts/download_data.py --dataset tusz            # full corpus (~70 GB)
python scripts/download_data.py --dataset tusz --sample   # dev split only
```

The script wraps `rsync -auxvL --partial <user>@www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.3/ data/tusz/` and feeds the password through `sshpass` if available (otherwise rsync prompts). Annotations: `*.csv_bi` (bi-class term-based, use these), `*.csv` (per-channel multi-class).

## 4. Helsinki neonatal EEG (open, Zenodo)

Stevenson et al. (2019) "A dataset of neonatal EEG recordings with seizure annotations", Zenodo. Find the record by searching Zenodo for the title (the concept record resolves to the latest version), then:

```bash
export HELSINKI_ZENODO_RECORD=<numeric record id>
python scripts/download_data.py --dataset helsinki            # all 79 EDFs + annotations
python scripts/download_data.py --dataset helsinki --sample   # first 2 EDFs + annotations
```

The script uses the Zenodo REST API (`https://zenodo.org/api/records/<id>`) to list files and downloads each with resume support. Annotations are per-second binary matrices (one column per neonate) for annotators A, B, C.

## 5. Manifests

After downloading, build the unified manifests (subject id, age, fs, channels, events):

```bash
python scripts/download_data.py --build-manifests
```

`manifests/records.csv` columns: `dataset, record_id, subject_id, path, fs, n_channels, duration_s, age_years`.
`manifests/events.csv` columns: `dataset, record_id, onset_s, offset_s, annotator`.

## Credentials and hygiene

- TUSZ credentials only via `TUH_USERNAME` / `TUH_PASSWORD` env vars (or a `.env` file that is git-ignored).
- Check free space before the full downloads (`df -h data`).
- Verify PhysioNet downloads with the provided `SHA256SUMS.txt` in each PhysioNet project folder.
