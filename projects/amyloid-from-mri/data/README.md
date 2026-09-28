# Data acquisition

Nothing in this folder is committed. All datasets below require registration; two require an application. Credentials are read from environment variables by `scripts/download_data.py`.

## 1. OASIS-3 (NITRC-IR / XNAT central) — free DUA

1. Create a NITRC account at https://www.nitrc.org/ and request access to the OASIS-3 project: https://www.nitrc.org/projects/oasis3/ → "Data Use Agreement". Approval is usually same-day to a few days.
2. Log in to the NITRC Image Repository (XNAT): https://www.nitrc.org/ir/ and open project `OASIS3`.
3. Download the spreadsheets. In the XNAT project page choose the "OASIS3_data_files" download (subject demographics, ADRC clinical data, MR sessions, PET sessions, PUP time-course tables, FreeSurfer list). Put them under `data/oasis3/tables/`. Typical file names (they vary slightly by release):
   - `OASIS3_demographics.csv` / `ADRC_ADRCCLINICALDATA.csv` (columns such as `ADRC_ADRCCLINICALDATA ID`, `Subject`, `ageAtEntry`, `mmse`, `cdr`, `apoe`, `M/F`)
   - `OASIS3_MR_sessions.csv` (`MR ID` like `OAS30001_MR_d0129`, `Scanner`)
   - `OASIS3_PUP*.csv` (`PUP_PUPTIMECOURSEDATA ID` like `OAS30001_AV45_PUPTIMECOURSE_d2430`, `tracer`, `Centiloid_fSUVR_TOT_CORTMEAN`)
   - `OASIS3_FreeSurfer*.csv` (`FS ID` like `OAS30001_Freesurfer53_d0129`)
4. Bulk downloads use the official scripts (https://github.com/NrgXnat/oasis-scripts):
   - `download_oasis_scans.sh <csv with experiment_id> <out_dir> <nitrc_user> [scan_type]` for MR sessions (you are prompted for the password).
   - `download_oasis_freesurfer.sh <csv with freesurfer_id> <out_dir> <nitrc_user>` for FreeSurfer 5.3 outputs (`aseg.stats`, `?h.aparc.stats`).
   - `download_oasis_pup.sh <csv with pup_id> <out_dir> <nitrc_user>` for PET Unified Pipeline outputs.
   The Python downloader in this repo does the same through the XNAT REST API with `OASIS_USER` / `OASIS_PASSWORD` (aliases `NITRC_USER` / `NITRC_PASSWORD`) and supports a `--sample N` mode.
5. Expected layout:

```
data/oasis3/
  tables/                         # spreadsheets from the NITRC download tab
  freesurfer/OAS30001_Freesurfer53_d0129/stats/{aseg.stats,lh.aparc.stats,rh.aparc.stats}
  mr/OAS30001_MR_d0129/anat1/NIFTI/*.nii.gz
  pup/OAS30001_AV45_PUPTIMECOURSE_d2430/...
```

## 2. ADNI (LONI IDA) — application

1. Apply at https://adni.loni.usc.edu/data-samples/access-data/ (institutional affiliation and a short project description; approval takes days to a couple of weeks).
2. Log in to https://ida.loni.usc.edu/, project ADNI → "Download → Study Data". Download:
   - `ADNIMERGE.csv` (age, sex, education, APOE4, MMSE, diagnosis per visit)
   - `UCBERKELEY_AMY_6MM*.csv` (amyloid PET SUVR and Centiloid; florbetapir and florbetaben)
   - `UCSFFSX*.csv` (FreeSurfer cross-sectional ROI tables; note the FreeSurfer version column)
   - `APOERES.csv` if APOE is not in ADNIMERGE for your release
3. For the CNN arm: "Download → Image Collections", filter `MPRAGE`/`Accelerated Sagittal MPRAGE`, create a collection, and download NIfTI with the IDA download manager. Keep the `*_MRILIST.csv` for image-to-visit matching.
4. Layout: `data/adni/tables/*.csv`, `data/adni/mri/<PTID>/<ImageUID>.nii.gz`.

ADNI data may not be redistributed or shared with anyone not covered by your DUA.

## 3. A4 / LEARN pre-randomization data (LONI IDA) — application

1. On https://ida.loni.usc.edu/ request access to the "A4" project (A4 Study pre-randomization data are shared publicly to approved investigators).
2. Download the screening/pre-randomization tables (demographics, APOE, PACC, `SUBJINFO`, amyloid PET SUVr/Centiloid) and the T1 MRI of screen-eligible participants (A+ in A4, A- in LEARN).
3. Layout: `data/a4/tables/*.csv`, `data/a4/mri/<BID>/*.nii.gz`.

## 4. Optional: OASIS-4 (NITRC-IR)

Same NITRC DUA process, project `OASIS4` at https://www.nitrc.org/projects/oasis4/. It has clinical, FreeSurfer and CSF data but no PET.

## Secrets

Never put credentials in the repo. Use environment variables:

```
export OASIS_USER=...        # NITRC username
export OASIS_PASSWORD=...    # NITRC password
```

`.env` files are git-ignored.
