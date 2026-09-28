# Data acquisition — Ped-ECG-Gen

Nothing here is committed. All primary datasets are open (no credentialing).

## 1. ZZU-pECG (pediatric target)

Published in *Scientific Data* 2025 (doi:10.1038/s41597-025-05225-z); the data are hosted on figshare (follow the "Data availability" link in the paper to the current figshare DOI). Download the WFDB records and the metadata/label CSV.

```bash
# after locating the figshare record URL from the paper's data-availability section:
python scripts/download_data.py --dataset zzu-pecg --url "<FIGSHARE_ARTICLE_URL>" --out data/zzu-pecg
```
Used: 12-lead WFDB records; metadata with age (years), sex, ECG diagnostic statements and ICD-10 disease labels.

## 2. Adult sources (open, PhysioNet / Zenodo)

```bash
python scripts/download_data.py --dataset ptbxl --out data/ptbxl --sample
python scripts/download_data.py --dataset challenge2021 --out data/challenge2021 --sample
python scripts/download_data.py --dataset code15 --out data/code15 --sample     # Zenodo REST API
```

## 3. Pediatric normal limits

Rijnbeek et al. 2001 (Eur Heart J 22:702-711) age-banded normal ranges are transcribed into `src/ped_ecg/pediatric_norms.py` (a small reference table); no download needed.

## Expected layout

```
data/
  zzu-pecg/
    metadata.csv          # child_id, age_years, sex, statements, icd10
    records/...           # WFDB .hea/.dat (12-lead)
  ptbxl/
    ptbxl_database.csv  scp_statements.csv  records500/...
  challenge2021/
    dx_mapping_scored.csv  training/...
  code15/
    exams.csv  exams_part*.hdf5
```

Notes: only 12-lead ZZU-pECG records are used for cross-dataset work; 9-lead records are excluded. CODE-15% is 400 Hz and resampled to 500 Hz by the loader.
