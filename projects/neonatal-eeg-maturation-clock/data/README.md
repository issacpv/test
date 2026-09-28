# Data acquisition

Nothing in this folder is committed except this file. All commands run from the project root. Expected total: ~15 GB.

## Expected layout

```
data/
  helsinki/                    # Stevenson et al. (2019), Zenodo (open)
    eeg1.edf ... eeg79.edf
    annotations_2017_A.csv annotations_2017_B.csv annotations_2017_C.csv
    clinical_information.csv
  hie_grading/                 # O'Toole et al. (2023), Zenodo (open)
    <as distributed: EDF/MAT epochs + grading table>
  manifests/                   # produced by scripts (safe to commit)
    records.csv  events.csv
```

## 1. Helsinki neonatal EEG (open, Zenodo)

Stevenson, Tapani, Lauronen & Vanhatalo (2019), "A dataset of neonatal EEG recordings with seizure annotations", *Sci. Data*. Find the numeric Zenodo record id (search the title on zenodo.org; the repository catalogue `DATASETS.md` lists record 2547147; the concept record resolves to the latest version), then:

```bash
export HELSINKI_ZENODO_RECORD=2547147
python scripts/download_data.py --dataset helsinki --sample   # 2 EDFs + annotations + clinical info
python scripts/download_data.py --dataset helsinki            # all 79 EDFs (~4 GB)
```

The script lists the record's files through `https://zenodo.org/api/records/<id>` and downloads each with resume support. PMA at recording is derived from the gestational-age and postnatal-age fields of `clinical_information.csv` (check the column names in the file; `--build-manifests` prints the columns it found and leaves `pma_weeks` empty if it cannot derive it).

## 2. Neonatal EEG graded for HIE background severity (open, Zenodo)

O'Toole et al. (2023), "Neonatal EEG graded for severity of background abnormalities in hypoxic-ischaemic encephalopathy", *Sci. Data*. Either set the record id directly or let the script search Zenodo:

```bash
export HIE_ZENODO_RECORD=<numeric record id>          # preferred
python scripts/download_data.py --dataset hie --sample
# or search (uses the Zenodo search API with pagination and picks the best title match):
python scripts/download_data.py --dataset hie --search
```

Keep the grading table exactly as distributed; `--build-manifests` joins grades to record ids by file name.

## 3. Manifests

```bash
python scripts/download_data.py --build-manifests
```

Writes `records.csv` (dataset, record_id, subject_id, path, fs, n_channels, duration_s, ga_weeks, pma_weeks, grade) and `events.csv` (dataset, record_id, onset_s, offset_s, annotator). Only public ids and clinical fields already released with the datasets are stored.
