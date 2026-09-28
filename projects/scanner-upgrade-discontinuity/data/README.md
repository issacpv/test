# Data acquisition

Nothing in this folder is committed. Both cohorts require a data-use agreement. Credentials are read from environment variables by `scripts/download_data.py`.

## 1. OASIS-3 (NITRC Image Repository / XNAT) — free DUA

1. Create a NITRC account (https://www.nitrc.org/), open https://www.nitrc.org/projects/oasis3/ and accept the OASIS Data Use Agreement. Approval is typically same day to a few days.
2. Log in to the NITRC-IR XNAT instance (https://www.nitrc.org/ir/) and open project `OASIS3`.
3. Download the spreadsheets from the project page ("OASIS3_data_files"). Put them in `data/oasis3/tables/`. The names vary slightly by release; the columns this project needs are:
   - MR sessions: `MR ID` (e.g. `OAS30001_MR_d0129`), `Subject`, `Scanner` (scanner model string; the project's first task is to tabulate its distinct values), `Scans` (sequence list).
   - FreeSurfer list: `FS ID` (e.g. `OAS30001_Freesurfer53_d0129`).
   - ADRC clinical data: `Subject`, `ageAtEntry`, days-from-entry in the row ID (`OAS30001_ClinicalData_d0000`), `cdr`, `sumbox` (CDR-SB), `mmse`, `apoe`, `M/F`.
4. Bulk downloads: the official scripts at https://github.com/NrgXnat/oasis-scripts (`download_oasis_freesurfer.sh`, `download_oasis_scans.sh`) or this repository's `scripts/download_data.py`, which uses the XNAT REST API with `OASIS_USER` / `OASIS_PASSWORD` (aliases `NITRC_USER` / `NITRC_PASSWORD`) and has a `--sample N` mode:

```
python scripts/download_data.py oasis-list       --out data/oasis3/tables --sample 50
python scripts/download_data.py oasis-freesurfer --ids data/oasis3/tables/freesurfer_ids.csv --out data/oasis3/freesurfer --sample 5
```

5. Traveling-subject subset: the 69 participants scanned on both the TIM Trio and the Biograph mMR within two weeks are identified from the MR session table (same subject, two sessions ≤ 14 days apart, different `Scanner`). `scanner_rd.sessions.find_paired_sessions` does this.
6. Expected layout:

```
data/oasis3/
  tables/                         # spreadsheets
  freesurfer/OAS30001_Freesurfer53_d0129/stats/{aseg.stats,lh.aparc.stats,rh.aparc.stats}
  mr/OAS30001_MR_d0129/anat1/NIFTI/*.nii.gz     # only for the optional image-level arm
```

OASIS-3 provides days from entry, not calendar dates. Do not try to reconstruct dates.

## 2. ADNI (LONI IDA) — application

1. Apply at https://adni.loni.usc.edu/data-samples/access-data/ (institutional affiliation, short project description; approval takes days to a few weeks).
2. Log in to https://ida.loni.usc.edu/, project ADNI → Download → Study Data. Download to `data/adni/tables/`:
   - `ADNIMERGE.csv` (RID, VISCODE, EXAMDATE, DX, CDRSB, MMSE, AGE, PTGENDER, APOE4, SITE)
   - `MRILIST.csv` (image-to-visit mapping, scan date, field strength, sequence name)
   - `MRIMETA.csv` / `MRI3META.csv` (scanner manufacturer and model per scan; 1.5T and 3T tables)
   - `UCSFFSL*.csv` (FreeSurfer longitudinal ROI tables; keep the FreeSurfer version column) and/or `UCSFFSX*.csv` (cross-sectional)
3. The ADNI-1 subset scanned at both 1.5T and 3T at the same visit is identified by matching `RID` + `VISCODE` across the 1.5T and 3T MRI tables within ±30 days.
4. Layout: `data/adni/tables/*.csv`. Imaging is only needed for the optional image-level arm (`data/adni/mri/<PTID>/<ImageUID>.nii.gz`).

ADNI data may not be shared with anyone not covered by your DUA, and manuscripts go through the ADNI Data and Publications Committee.

## 3. Optional: OASIS-4 (NITRC-IR)

Same NITRC DUA process, project `OASIS4` at https://www.nitrc.org/projects/oasis4/ (clinical cohort, FreeSurfer outputs, no PET).

## Secrets

```
export OASIS_USER=...        # NITRC username
export OASIS_PASSWORD=...    # NITRC password
```

Never commit credentials; `.gitignore` excludes `.env`, `*.netrc` and `.xnat_credentials`.
