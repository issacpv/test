# Data acquisition — tau-proxy-from-mri

Nothing in this folder is committed. All cohorts require a Data Use Agreement. Credentials are read from environment variables only.

## 1. OASIS-3 and OASIS-3_AV1451 (NITRC / XNAT Central)

1. Create a NITRC account: https://www.nitrc.org/account/register.php
2. Request access to the OASIS-3 project (https://www.nitrc.org/projects/oasis3/) and, separately, to the tau sub-project OASIS-3_AV1451 (https://www.nitrc.org/projects/oasis3_av1451/) and, if you want conversions, OASIS3_AV1451_Longitudinal (https://www.nitrc.org/projects/oasis3_av1451l/). Sign the DUA in each; approval is manual (days).
3. Once approved, data are served by XNAT Central (https://central.xnat.org). Export credentials:

   ```bash
   export XNAT_USER="your_nitrc_username"
   export XNAT_PASS="your_nitrc_password"
   ```

4. Download the tabular files first (small): from the OASIS-3 XNAT project, the "OASIS3_data_files" resource contains `OASIS3_demographics.csv`, `OASIS3_UDSb4_cdr.csv`, `OASIS3_UDSb9_...`, `OASIS3_FreeSurfer_*.csv` (aseg/aparc summaries), `OASIS3_PUP_*.csv` (amyloid PUP ROI tables, Centiloids) and, in the AV1451 project, `OASIS3_AV1451_PUP*.csv` (tau ROI SUVRs). The exact file names change between releases; list the resource with:

   ```bash
   python scripts/download_data.py --list-resources --project OASIS3
   python scripts/download_data.py --list-resources --project OASIS3_AV1451
   ```

5. Download tables:

   ```bash
   python scripts/download_data.py --tables --project OASIS3
   python scripts/download_data.py --tables --project OASIS3_AV1451
   ```

6. Download imaging only for the subjects with tau PET (`--subjects-from data/oasis3/tau_subjects.txt`, which `tau_proxy.tables` writes). Per-session FreeSurfer outputs (`FS` assessors) and FLAIR NIfTIs are fetched with:

   ```bash
   python scripts/download_data.py --freesurfer --project OASIS3 --subjects-from data/oasis3/tau_subjects.txt
   python scripts/download_data.py --flair --project OASIS3 --subjects-from data/oasis3/tau_subjects.txt
   ```

   The scripted downloads use the XNAT REST API (`/data/projects/{project}/subjects/{subject}/experiments/{experiment}/resources/...`). The official alternative is the `oasis-scripts` repository (https://github.com/NrgXnat/oasis-scripts) with `download_oasis_scans.sh` / `download_freesurfer.sh`; either route works.

Expected layout:

```
data/oasis3/
  tables/OASIS3_demographics.csv
  tables/OASIS3_UDSb4_cdr.csv
  tables/OASIS3_FreeSurfer_aseg.csv          # or per-session stats below
  tables/OASIS3_PUP_PIB.csv, OASIS3_PUP_AV45.csv
  tables/OASIS3_AV1451_PUP.csv
  freesurfer/OAS30001_MR_d0129/stats/aseg.stats, lh.aparc.stats, rh.aparc.stats
  flair/OAS30001_MR_d0129/sub-OAS30001_ses-d0129_FLAIR.nii.gz
```

## 2. ADNI (LONI IDA)

1. Apply at https://adni.loni.usc.edu/data-samples/access-data/ (institutional affiliation, DUA). Approval takes ~1-2 weeks.
2. In IDA (https://ida.loni.usc.edu), download from "Study Data":
   - `ADNIMERGE.csv` (age, sex, education, APOE4, MMSE, CDRSB, DX, scanner).
   - UC Berkeley tau PET tables: `UCBERKELEY_TAU_6MM_*.csv` (or the latest `UCBERKELEYAV1451_*`), which include META_TEMPORAL and Braak ROI SUVRs with inferior-cerebellar-GM reference and PVC variants.
   - UC Berkeley amyloid tables (`UCBERKELEY_AMY_6MM_*.csv`) for Centiloids.
   - UCSF FreeSurfer cross-sectional tables (`UCSFFSX6_*.csv`, `UCSFFSX7_*.csv`) for thickness and volumes; alternatively download raw T1/FLAIR and run FreeSurfer 7 yourself.
3. Place the CSVs in `data/adni/tables/`. There is no scripted download: IDA requires an interactive login and the DUA prohibits redistribution. `scripts/download_data.py --adni-check` only verifies that the expected files are present and prints row counts.

## 3. A4 / LEARN (optional)

Apply through IDA (project "A4"). Tau PET is available only for a subset; use the screening tau tables and the `A4_PETSUVR` files. Same placement rule: `data/a4/tables/`.

## 4. Sample mode (no credentials)

`python scripts/download_data.py --sample` writes a small synthetic OASIS-3-like table set to `data/sample/` so that the pipeline (`tau_proxy.tables` → `labels` → `features` → `models`) can be exercised end-to-end without any DUA data.
