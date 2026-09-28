# Data acquisition — fmri-hemodynamic-aging

Nothing in this directory is committed. All downloads land under `data/` (git-ignored). Credentials come from environment variables; never write them into files in the repo.

## 1. HCP-Aging (NIMH Data Archive; free with a Data Use Certification)

1. Create an NDA account at https://nda.nih.gov/ and request access to the "Lifespan Human Connectome Project Aging" collection (collection ID 2847). Your institution's signing official must approve the Data Use Certification; allow 1-3 weeks.
2. Install the NDA tools: `pip install nda-tools`.
3. In the NDA web interface, filter the collection to the image-processed packages you need and create a package. Minimal set for this project:
   - `rfMRI_REST*_AP/PA` minimally preprocessed volume series (`*_rfMRI_REST1_AP.nii.gz`, motion regressors, `Physio_log.txt`),
   - `mbPCASLhr` (multi-PLD pCASL) raw plus the HCP-ASL derivative package if available (`perfusion_calib.nii.gz`, `arrival.nii.gz`),
   - `T1w_restore.nii.gz`, `T2w_restore.nii.gz`, `aparc+aseg`,
   - phenotypic instruments: `ndar_subject01`, `vitals01` (blood pressure), `bsc01`/`bloodlabs` (HbA1c, lipids), `medications`, `ssaga` (smoking).
4. Download with `downloadcmd` (username/password from env vars):

```bash
export NDA_USERNAME=... ; export NDA_PASSWORD=...
downloadcmd -dp <package_id> -d data/hcp_aging -u "$NDA_USERNAME" -p "$NDA_PASSWORD" -wt 8
```

`scripts/download_data.py --source hcp-aging --package-id <id>` wraps this call and refuses to run if the env vars are missing.

Expected layout:

```
data/hcp_aging/
  HCA<subject>_V1_MR/
    MNINonLinear/Results/rfMRI_REST1_AP/rfMRI_REST1_AP.nii.gz
    MNINonLinear/Results/rfMRI_REST1_AP/Movement_Regressors.txt
    MNINonLinear/Results/rfMRI_REST1_AP/rfMRI_REST1_AP_Physio_log.txt
    ... REST1_PA, REST2_AP, REST2_PA ...
    T1w/T1w_acpc_dc_restore.nii.gz
    ASL/  (or derivatives from the HCP-ASL pipeline)
  phenotypes/*.txt
```

## 2. OASIS-3 (NITRC / XNAT Central; free registration + DUA)

1. Register at https://www.nitrc.org/, request OASIS-3 access (https://www.nitrc.org/projects/oasis3/) and sign the DUA. Approval usually takes days.
2. Use the OASIS download scripts (https://github.com/NrgXnat/oasis-scripts) or the XNAT REST API. `scripts/download_data.py --source oasis3 --session-list <csv>` implements the REST route with `requests` and reads `NITRC_USER` / `NITRC_PASS`.
3. Session lists (which MR sessions have rs-fMRI, FLAIR, and a PET session within ±1 year) are built from the OASIS-3 CSV exports on the XNAT "Data" tab: `OASIS3_data_files/MR_sessions.csv`, `PET_sessions.csv`, `UDS/*.csv`.

Expected layout (BIDS-like, as delivered):

```
data/oasis3/
  sub-OAS30001/ses-d0129/func/sub-OAS30001_ses-d0129_task-rest_run-01_bold.nii.gz
  sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_T1w.nii.gz
  sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_FLAIR.nii.gz
  clinical/UDS_*.csv, pup/*.csv (Centiloids), freesurfer/*.csv
```

## 3. Open smoke-test data — OpenNeuro ds000030 (CC0)

```bash
python scripts/download_data.py --source openneuro --dataset ds000030 --sample   # 2 subjects, rest + T1w only
# or: pip install openneuro-py && openneuro-py download --dataset ds000030 --include 'sub-1000[12]/*'
# or: aws s3 sync --no-sign-request s3://openneuro.org/ds000030/sub-10001 data/openneuro/ds000030/sub-10001
```

## 4. Optional — NKI-Rockland enhanced (free DUA)

Raw BIDS imaging is on the public `fcp-indi` S3 bucket; phenotypes require the NKI DUA (https://fcon_1000.projects.nitrc.org/indi/enhanced/).

```bash
aws s3 ls --no-sign-request s3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest/
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest/sub-A00008326 data/nki/sub-A00008326
```

## 5. Atlases

- Schaefer-400 (7-network order) MNI 2 mm and Tian S2 subcortical parcellation: `nilearn.datasets.fetch_atlas_schaefer_2018`; Tian atlas from https://github.com/yetianmed/subcortex (MNI152NLin2009cAsym).
- Venous atlas / sinus mask: derive from HCP-Aging T2*-weighted data or use a published venous atlas if licensed for reuse.

## Derivatives produced by this project

```
outputs/
  lag/<dataset>/<sub>_<ses>_<run>_lag.nii.gz, *_corr.nii.gz, *_parcels.tsv
  hrf/<dataset>/<sub>_<ses>_<run>_hrfparams.tsv
  asl/<sub>_att_parcels.tsv
  tables/features.parquet, normative_params.json, hemodynamic_age.tsv
```
