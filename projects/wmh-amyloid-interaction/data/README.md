# Data acquisition

Nothing here is committed. OASIS-3 needs a free DUA; ADNI an application. Credentials are read from environment variables by `scripts/download_data.py`.

## 1. OASIS-3 (NITRC-IR / XNAT) — free DUA

1. NITRC account (https://www.nitrc.org/) → project https://www.nitrc.org/projects/oasis3/ → accept the Data Use Agreement.
2. Log in to https://www.nitrc.org/ir/, project `OASIS3`. Download the spreadsheets ("OASIS3_data_files") into `data/oasis3/tables/`:
   - MR sessions (`MR ID`, `Subject`, `Scanner`, scan list including `FLAIR`, `T1w`),
   - PUP time-course table (`PUP_PUPTIMECOURSEDATA ID` like `OAS30001_PIB_PUPTIMECOURSE_d2430`, `tracer`, `Centiloid_fSUVR_TOT_CORTMEAN`),
   - ADRC clinical data (`Subject`, `ageAtEntry`, `cdr`, `sumbox`, `mmse`, `apoe`, `M/F`, row IDs `OAS30001_ClinicalData_d0000`),
   - psychometrics (`OAS30001_UDSb_d0000`-style IDs: logical memory, digit symbol, animals, Trails, Boston naming),
   - FreeSurfer list (`FS ID`, for eTIV / ventricles / WM-hypointensities).
3. Imaging downloads (FLAIR + T1 NIfTI per session; FreeSurfer stats) with the official scripts (https://github.com/NrgXnat/oasis-scripts: `download_oasis_scans.sh <csv> <out> <user> FLAIR`) or with this repository's downloader (XNAT REST API, `OASIS_USER` / `OASIS_PASSWORD`, `--sample N`):

```
python scripts/download_data.py oasis-list  --out data/oasis3/tables --sample 50      # sessions + which have FLAIR
python scripts/download_data.py oasis-flair --ids data/oasis3/tables/flair_sessions.csv --out data/oasis3/mr --sample 3
python scripts/download_data.py oasis-freesurfer --ids data/oasis3/tables/freesurfer_ids.csv --out data/oasis3/freesurfer --sample 3
```

4. Expected layout:

```
data/oasis3/
  tables/
  mr/OAS30001_MR_d0129/{anat2,anat3}/NIFTI/*.nii.gz     # FLAIR and T1 scan folders (names vary)
  freesurfer/OAS30001_Freesurfer53_d0129/stats/aseg.stats
  wmh/<tool>/OAS30001_MR_d0129_wmh_mask.nii.gz          # your segmentations (never committed)
```

OASIS-3 releases days-from-entry, not calendar dates. The `Scanner` column identifies 1.5T (Vision, Sonata) vs 3T (TIM Trio, Biograph mMR, ...) sessions.

## 2. ADNI (LONI IDA) — application

1. Apply at https://adni.loni.usc.edu/data-samples/access-data/.
2. IDA → Download → Study Data → into `data/adni/tables/`:
   - `ADNIMERGE.csv` (RID, VISCODE, EXAMDATE, DX, CDRSB, MMSE, AGE, PTGENDER, PTEDUCAT, APOE4, ICV),
   - UC Davis WMH volume tables (search the Study Data listing for "UCD" and "WMH"; ADNI-2/3 FLAIR-based WMH volumes with `TOTAL_WMH`, ICV and exam date),
   - `UCBERKELEY_AMY_6MM*.csv` (Centiloids; florbetapir/florbetaben),
   - `UCSFFSX*.csv` (FreeSurfer eTIV).
3. Raw FLAIR (only if re-running segmenters): Download → Image Collections → filter `FLAIR`/`Sagittal 3D FLAIR`; `data/adni/flair/<PTID>/<ImageUID>.nii.gz`.

## 3. WMH Segmentation Challenge (MICCAI 2017) — open with registration

https://wmh.isi.uu.nl/ — 60 training cases with manual masks (three scanners) to calibrate/validate the segmenters before applying them to OASIS-3. Layout: `data/wmh_challenge/<site>/<case>/{FLAIR.nii.gz,T1.nii.gz,wmh.nii.gz}`.

## Secrets

```
export OASIS_USER=...        # NITRC username
export OASIS_PASSWORD=...    # NITRC password
```
