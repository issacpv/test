# Data acquisition

No data are committed. OASIS-3 needs a free NITRC account + DUA; ADNI needs an application.

## 1. OASIS-3 (NITRC-IR / XNAT central) — free DUA

1. Register at https://www.nitrc.org/ and accept the OASIS-3 Data Use Agreement at https://www.nitrc.org/projects/oasis3/.
2. Open the XNAT project at https://www.nitrc.org/ir/ (project `OASIS3`).
3. Spreadsheets: download the "OASIS3_data_files" bundle from the project page and put it under `data/oasis3/tables/`. Needed:
   - ADRC clinical data (`ADRC_ADRCCLINICALDATA ID` = `OAS30001_ClinicalData_d0000`, `cdr`, `mmse`, `sumbox` (CDR sum of boxes), `ageAtEntry`, `apoe`, `M/F`)
   - MR sessions (`MR ID` = `OAS30001_MR_d0129`, `Scanner`)
   - PUP time-course table (`PUP_PUPTIMECOURSEDATA ID`, `tracer`, `Centiloid_fSUVR_TOT_CORTMEAN`)
   - FreeSurfer list (`FS ID` = `OAS30001_Freesurfer53_d0129`)
4. FreeSurfer outputs (`stats/aseg.stats`, `stats/?h.aparc.stats`): official script `download_oasis_freesurfer.sh <csv with freesurfer_id> <out_dir> <nitrc_user>` from https://github.com/NrgXnat/oasis-scripts, or `python scripts/download_data.py oasis-freesurfer ...` (reads `OASIS_USER`/`OASIS_PASSWORD`).
5. FLAIR (+ T1) for WMH segmentation: `download_oasis_scans.sh <csv with experiment_id> <out_dir> <nitrc_user> T1w,FLAIR` or `python scripts/download_data.py oasis-mr --scan-type T1w,FLAIR ...`. Not every MR session has FLAIR; the timeline builder marks sessions with/without it.
6. WMH segmentation (choose one; record the choice in `outputs/provenance.json`):
   - FreeSurfer 7: `run_samseg --input T1.nii.gz FLAIR.nii.gz --lesion --lesion-mask-pattern 0 1 --output <dir>`; WMH volume from `samseg.stats` ("Lesions").
   - LST-AI (https://github.com/CompImg/LST-AI) on T1+FLAIR.
   Write a table `data/oasis3/wmh/wmh_volumes.csv` with columns `mr_id, wmh_mm3, method`.

Layout:

```
data/oasis3/
  tables/*.csv
  freesurfer/<FS ID>/stats/{aseg.stats,lh.aparc.stats,rh.aparc.stats}
  mr/<MR ID>/...                 # FLAIR/T1 NIfTI
  wmh/wmh_volumes.csv
```

## 2. ADNI (LONI IDA) — application

1. Apply at https://adni.loni.usc.edu/data-samples/access-data/.
2. From https://ida.loni.usc.edu/ → ADNI → Download → Study Data, download:
   - `ADNIMERGE.csv` (CDRSB, MMSE, DX, AGE, PTGENDER, APOE4, per-visit)
   - `UCBERKELEY_AMY_6MM*.csv` (Centiloid per PET)
   - `UCSFFSX*.csv` / `UCSFFSL*.csv` (FreeSurfer cross-sectional / longitudinal ROI volumes and thickness, with ICV)
   - `UCD_WMH*.csv` (UC Davis white-matter-hyperintensity volumes from FLAIR)
3. Layout: `data/adni/tables/*.csv`. Imaging is not required if the ROI/WMH tables are used.

## 3. Optional: OASIS-4 (NITRC-IR)

Project `OASIS4`, same DUA route (https://www.nitrc.org/projects/oasis4/). No PET; CSF Aβ42/40 on a subset can define amyloid status for a memory-clinic replication.

## 4. pySuStaIn

```
pip install git+https://github.com/ucl-pond/pySuStaIn.git
```

## Secrets

```
export OASIS_USER=...      # NITRC username
export OASIS_PASSWORD=...
```
Never commit credentials.
