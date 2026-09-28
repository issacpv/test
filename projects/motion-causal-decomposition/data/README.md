# Data acquisition

Nothing here is committed. ABIDE is fully open; HCP needs a free registration; ABCD needs an NDA application.

## 1. ABIDE I / II preprocessed (Preprocessed Connectomes Project) — open

The PCP hosts denoised ROI time series and a phenotypic file with per-subject mean framewise displacement (`func_mean_fd`) on a public S3 bucket (anonymous HTTPS):

- Phenotypic file: `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv`
- ROI time series: `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/<pipeline>/<strategy>/rois_<atlas>/<FILE_ID>_rois_<atlas>.1D` with `pipeline ∈ {cpac, ccs, dparsf, niak}`, `strategy ∈ {filt_global, filt_noglobal, nofilt_global, nofilt_noglobal}`, `atlas ∈ {aal, cc200, cc400, dosenbach160, ez, ho, tt}`.

`scripts/download_data.py abide --sample 20` downloads the phenotypic file and the first 20 CC200 series for two strategies (with/without global signal regression). ABIDE II raw/preprocessed data: http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html (free NITRC registration for some sites).

Layout:

```
data/abide/
  Phenotypic_V1_0b_preprocessed1.csv
  cpac/filt_noglobal/rois_cc200/<FILE_ID>_rois_cc200.1D
  cpac/filt_global/rois_cc200/<FILE_ID>_rois_cc200.1D
```

## 2. HCP Young Adult S1200 — free registration

1. Register at https://db.humanconnectome.org, accept the Open Access Data Use Terms.
2. For S3 access: in ConnectomeDB, enable "Amazon S3 access" to obtain an AWS key pair for the `hcp-openaccess` bucket. Export them as `HCP_AWS_ACCESS_KEY_ID` / `HCP_AWS_SECRET_ACCESS_KEY`.
3. Per subject and resting run (`rfMRI_REST1_LR`, `rfMRI_REST1_RL`, `rfMRI_REST2_LR`, `rfMRI_REST2_RL`) this project needs:
   - `HCP_1200/<subject>/MNINonLinear/Results/<run>/Movement_Regressors.txt` (12 columns: 3 translations mm, 3 rotations degrees, and their derivatives)
   - `HCP_1200/<subject>/MNINonLinear/Results/<run>/Movement_RelativeRMS_mean.txt`
   - node time series: either the HCP_PTN1200 group-ICA release (`3T_HCP1200_MSMAll_d100_ts2.tar.gz`, ConnectomeDB "Resting State fMRI 1200 Subjects Group-ICA" package) or the `*_Atlas_MSMAll_hp2000_clean.dtseries.nii` parcellated with your atlas of choice.
   - diffusion motion (negative-control exposure): `HCP_1200/<subject>/T1w/Diffusion/eddylogs/eddy_unwarped_images.eddy_movement_rms`.
4. Behavioural tables: download `unrestricted_*.csv` from ConnectomeDB (NIH Toolbox scores, `Movement_RelativeRMS_mean` per run is in the imaging tables). Family structure requires the Restricted Data application (`restricted_*.csv`, columns `Family_ID`, `Age_in_Yrs`).

`scripts/download_data.py hcp --sample 3` lists and fetches the movement files for 3 subjects (needs boto3 + keys).

Layout: `data/hcp/<subject>/<run>/Movement_Regressors.txt`, `data/hcp/tables/unrestricted.csv`, `data/hcp/tables/restricted.csv`, `data/hcp/ptn/…`.

## 3. ABCD Study — NDA application

1. Obtain an NDA account and a Data Use Certification for the ABCD Study collection (https://nda.nih.gov/abcd). Institutional signing official required; weeks.
2. From the NDA, download the ABCD-BIDS derivatives or the DCAN connectomes for the resting runs, the run-level motion QC tables (`mri_y_qc_motion`), the imaging session/scanner table (`mri_y_adm_info`), NIH Toolbox (`nc_y_nihtb`) and CBCL (`mh_p_cbcl`) tables. Table names follow the release-5 naming; check the release notes.
3. Layout: `data/abcd/tables/*.csv`, `data/abcd/connectomes/<sub>/<ses>/<run>.npy`.

## 4. CoRR — open

http://fcon_1000.projects.nitrc.org/indi/CoRR/html/ (test-retest sessions; used only for run-level artifact sensitivity). Download via the INDI S3 bucket (`s3://fcp-indi/data/Projects/CoRR/`) analogously to ABIDE.

## Secrets

```
export HCP_AWS_ACCESS_KEY_ID=...
export HCP_AWS_SECRET_ACCESS_KEY=...
```

Never commit keys; `.gitignore` excludes `.env`, `*.netrc` and `.aws/`.
