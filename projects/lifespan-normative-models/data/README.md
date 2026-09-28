# Data acquisition

Nothing under `data/` is committed. Four access routes are involved: ConnectomeDB (HCP-YA), NDA (HCP-A, HCP-D), NITRC-IR (OASIS-3) and OpenNeuro (open). Credentials go in environment variables read by `scripts/download_data.py`.

## 1. HCP Young Adult (ConnectomeDB) — free registration + data use terms

1. Register at https://db.humanconnectome.org and accept the WU-Minn HCP Open Access Data Use Terms.
2. Behavioural/demographic table: in ConnectomeDB open "WU-Minn HCP Data - 1200 Subjects" → "Open Access" → download `unrestricted_<user>_<date>.csv` (columns `Subject`, `Gender`, `Age` as bins `22-25`, `26-30`, `31-35`, `36+`). Save as `data/hcp_ya/tables/unrestricted.csv`. Exact ages, family IDs, etc. are in the Restricted Data file, which requires a separate application; if granted, save it as `data/hcp_ya/tables/restricted.csv` and never commit or share it.
3. Imaging via AWS S3: in ConnectomeDB, enable "Amazon S3 Access" to obtain an access key/secret for the bucket `s3://hcp-openaccess`. Export them as `HCP_AWS_ACCESS_KEY_ID` / `HCP_AWS_SECRET_ACCESS_KEY`. Then:

```
python scripts/download_data.py hcp-ya --what freesurfer --sample 5      # <subj>/T1w/<subj>/stats/*.stats
python scripts/download_data.py hcp-ya --what diffusion  --sample 2      # <subj>/T1w/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}
```
   (equivalently `aws s3 cp s3://hcp-openaccess/HCP_1200/<subj>/T1w/Diffusion/ ... --recursive`). The 45-subject retest release is under `HCP_Retest/`.

## 2. HCP-Aging and HCP-Development (NDA) — account + Data Use Certification

1. Create an NDA account (https://nda.nih.gov/), then request access to the HCP Lifespan permission group (Data Use Certification signed by you and an institutional official). HCP-D is NDA collection 2846, HCP-A is 2847.
2. In NDA, build a Data Package: "Data from Labs" → HCP Aging / HCP Development → Lifespan 2.0 Release → select imaging (`imagingcollection01`, preprocessed structural/diffusion) and phenotypic (`ndar_subject01`, demographics with `interview_age` in months, `sex`) → "Create Package" (note the package id).
3. Install `nda-tools` (`pip install nda-tools`) and download:

```
export NDA_USERNAME=... NDA_PASSWORD=...
python scripts/download_data.py hcp-lifespan --package-id <id> --out data/hcp_lifespan
# wraps: downloadcmd -dp <id> -u $NDA_USERNAME -p $NDA_PASSWORD -d data/hcp_lifespan
```
   Alternatively use the S3 links in the package manifest (`downloadcmd -dp <id> -s3 ...`).

## 3. OASIS-3 (NITRC-IR / XNAT) — free DUA

Same procedure as in the OASIS documentation: NITRC account → accept OASIS-3 DUA (https://www.nitrc.org/projects/oasis3/) → XNAT project `OASIS3` at https://www.nitrc.org/ir/. Download the spreadsheets (`OASIS3_data_files`) to `data/oasis3/tables/` and use either the official scripts (https://github.com/NrgXnat/oasis-scripts: `download_oasis_freesurfer.sh`, `download_oasis_scans.sh ... dwi`) or:

```
export OASIS_USER=... OASIS_PASSWORD=...
python scripts/download_data.py oasis3 --what freesurfer --ids data/oasis3/tables/freesurfer_ids.csv --sample 5
python scripts/download_data.py oasis3 --what dwi --ids data/oasis3/tables/mr_ids.csv --sample 2
```

## 4. OpenNeuro (open, no credentials)

Datasets: `ds004169` (QTIM: 1,202 healthy 12-30 y from 682 families; T1w with two acquisition protocols, DWI on ~690 at session 1, second session for ~140 with 58 repeat DWI; columns `family_id, sex, age, age_ses02, ses01_T1w_acq, ses02_T1w_acq, ses01_dwi_acq, ses02_dwi_acq`), `ds000030` (UCLA CNP: 130 controls, 50 SCZ, 49 BD, 43 ADHD; T1 + 64-dir DWI; sex is in the `gender` column, diagnosis in `diagnosis`), `ds000221` (MPI-LEMON, T1 + DWI, 20-80 y), `ds003097` (AOMIC-ID1000, T1 + DWI, young adults).

```
python scripts/download_data.py openneuro --datasets ds004169 ds000030 --sample 3
```
fetches `participants.tsv` + `dataset_description.json` from the public bucket (https://s3.amazonaws.com/openneuro.org/<ds>/...) and, if `aws` or `openneuro-py` is installed, the first N subjects. Full download alternatives:

```
aws s3 sync --no-sign-request s3://openneuro.org/ds000030 data/openneuro/ds000030
openneuro-py download --dataset ds000030 --target-dir data/openneuro/ds000030
datalad install https://github.com/OpenNeuroDatasets/ds000030.git
```

## 5. Reference curves (open)

```
python scripts/download_data.py references     # git clone brainchart/Lifespan and predictive-clinical-neuroscience/braincharts
```

## Expected layout

```
data/
  hcp_ya/tables/unrestricted.csv          data/hcp_ya/HCP_1200/<subj>/T1w/...
  hcp_lifespan/<package files>            # ndar_subject01.txt, imagingcollection01.txt, image files
  oasis3/tables/*.csv                     data/oasis3/freesurfer/<FS ID>/stats/*.stats   data/oasis3/dwi/<MR ID>/
  openneuro/<ds>/participants.tsv         data/openneuro/<ds>/sub-*/
  references/Lifespan/                    data/references/braincharts/
  derived/features_thickness.csv  features_volumes.csv  features_fa.csv  features_md.csv  phenotypes.csv
```

## Secrets

`HCP_AWS_ACCESS_KEY_ID`, `HCP_AWS_SECRET_ACCESS_KEY`, `NDA_USERNAME`, `NDA_PASSWORD`, `OASIS_USER`, `OASIS_PASSWORD`. Never commit them; `.env` is git-ignored.
