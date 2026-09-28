# Data acquisition

Nothing in this folder is committed except this file. All commands run from the project root. Expected total: ~60 GB.

## Expected layout

```
data/
  vitaldb/
    cases.csv                  # clinical information table (api.vitaldb.net/cases)
    trks.csv                   # track list: caseid, tname, tid (api.vitaldb.net/trks)
    tracks/<caseid>/<tname>.csv.gz   # one file per downloaded track (Time, value)
    case_lists/propofol.csv sevoflurane.csv   # produced by --build-manifests
  cambridge_propofol/          # Chennu et al. (2016) sedation EEG (Apollo repository)
    <subject>/... (.set/.fdt or .mat as distributed) + behavioural / plasma tables
  manifests/
    cases.csv  epochs_index.csv
```

## 1. VitalDB open dataset (open, free)

VitalDB serves its open dataset through a REST API (no key needed) and the `vitaldb` Python package. Three endpoints are used:

- `https://api.vitaldb.net/cases` - clinical information for all cases (CSV; columns include `caseid`, `subjectid`, `age`, `sex`, `height`, `weight`, `asa`, `ane_type`, `casestart`, `caseend`, `anestart`, `aneend`, `opstart`, `opend`, and drug totals such as `intraop_ppf`).
- `https://api.vitaldb.net/trks` - list of available tracks per case (CSV; `caseid`, `tname`, `tid`).
- `https://api.vitaldb.net/<tid>` - the data of one track (CSV; `Time`, value column).

```bash
python scripts/download_data.py --dataset vitaldb --sample      # tables + 3 propofol and 3 sevoflurane cases
python scripts/download_data.py --dataset vitaldb                # all eligible general-anaesthesia cases with EEG
python scripts/download_data.py --build-manifests                # writes case_lists/*.csv and manifests/cases.csv
```

Tracks fetched per case: `BIS/EEG1_WAV`, `BIS/EEG2_WAV`, `BIS/BIS`, `BIS/SEF`, `BIS/SR`, `BIS/EMG`, `BIS/SQI`, `Orchestra/PPF20_CE`, `Orchestra/PPF20_RATE`, `Orchestra/RFTN20_CE`, `Primus/EXP_SEVO`, `Primus/INSP_SEVO`, `Primus/EXP_DES`, `Primus/MAC`, `Solar8000/HR`, `Solar8000/ART_MBP`. Track names are matched case-insensitively against `trks.csv`; missing tracks are recorded as absent, not errors.

Equivalent with the package: `pip install vitaldb` then `vitaldb.load_case(caseid, ["BIS/EEG1_WAV", "BIS/BIS"], 1/128)`. The website's bulk downloader (vitaldb.net/dataset) requires registration and acceptance of the data-use terms; the API serves the same open data.

## 2. Cambridge propofol sedation EEG (open)

Chennu et al. (2016), *PLoS Comput. Biol.*, "Brain connectivity dissociates responsiveness from sedation during anesthetic-induced unconsciousness". The data are deposited in the University of Cambridge Apollo repository (https://www.repository.cam.ac.uk/); search for the paper title or "propofol sedation EEG Chennu", note the record's licence, and copy the download URL(s):

```bash
export CAMBRIDGE_PROPOFOL_URL="https://www.repository.cam.ac.uk/....../download"   # one or more, comma-separated
python scripts/download_data.py --dataset cambridge
```

Files are EEGLAB/MATLAB exports per participant and sedation level, with plasma propofol concentrations and behavioural hit rates in accompanying tables. Only frontal channels are used here (see `doa_xfer.vitaldb_io.frontal_channel_picks`).

## 3. Manifests

`--build-manifests` writes `data/vitaldb/case_lists/propofol.csv` and `sevoflurane.csv` (caseid, subjectid, age, sex, asa, agent, has_eeg, tracks present) and `data/manifests/cases.csv`. These contain only VitalDB case identifiers and derived flags and may be committed.
