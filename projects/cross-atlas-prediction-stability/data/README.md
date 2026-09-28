# Data acquisition

Nothing here is committed. AOMIC and CoRR are open; HCP needs a free registration; two atlases need a free registration.

## 1. Atlases

`scripts/download_data.py atlases --out data/atlases` fetches with nilearn:

- Schaefer 2018 (100, 200, 400, 600, 800, 1000 parcels; 7 and 17 networks; MNI 2 mm) — `nilearn.datasets.fetch_atlas_schaefer_2018`
- AAL (SPM12 version) — `fetch_atlas_aal`
- Craddock 2012 (scorr/tcorr, multiple resolutions) — `fetch_atlas_craddock_2012`
- DiFuMo (64-1024 dimensions, soft) — `fetch_atlas_difumo`
- Harvard-Oxford cortical/subcortical — `fetch_atlas_harvard_oxford`
- Yeo 2011 7/17 network labels (for network aggregation) — `fetch_atlas_yeo_2011`

Surface (fsLR-32k) versions for HCP dense series:

- Schaefer fsLR dlabel files from the CBIG repository: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti
- Glasser MMP1.0 (`Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii`) from BALSA (free registration): https://balsa.wustl.edu/study/RVVG
- Gordon 333 parcels (surface and volume): https://sites.wustl.edu/petersenschlaggarlab/resources/ (or BALSA)
- Brainnetome 246 (volume and fsLR): https://atlas.brainnetome.org (free registration)

Layout: `data/atlases/<name>/...`; keep a `data/atlases/manifest.csv` with name, family, resolution, space.

## 2. AOMIC (OpenNeuro) — open

Datasets: ID1000 `ds003097`, PIOP1 `ds002785`, PIOP2 `ds002790`. fMRIPrep derivatives are included under `derivatives/fmriprep/`. Preprocessed MNI-space BOLD for resting state (PIOP1/2 `task-restingstate`; ID1000 has a movie-watching run used as "rest") and confounds:

```
python scripts/download_data.py aomic --dataset ds002785 --out data/aomic --sample 5
# or
aws s3 sync --no-sign-request s3://openneuro.org/ds002785 data/aomic/ds002785 \
    --exclude "*" --include "participants.tsv" --include "derivatives/fmriprep/sub-*/func/*restingstate*MNI152NLin2009cAsym*preproc_bold.nii.gz" --include "derivatives/fmriprep/sub-*/func/*restingstate*confounds*.tsv"
```

Phenotypes are in `participants.tsv` (age, sex, IQ scores, education, NEO-FFI in ID1000). Layout: `data/aomic/<dsid>/participants.tsv`, `data/aomic/<dsid>/derivatives/fmriprep/sub-XXXX/func/...`.

## 3. HCP Young Adult S1200 — free registration

1. Register at https://db.humanconnectome.org; accept the Open Access terms; enable S3 access and export `HCP_AWS_ACCESS_KEY_ID` / `HCP_AWS_SECRET_ACCESS_KEY`.
2. Dense series per run: `s3://hcp-openaccess/HCP_1200/<subject>/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii` (~1 GB each). Alternatively the HCP_PTN1200 package (ConnectomeDB → "Resting State fMRI 1200 Subjects Group-ICA" and node time series) if only ICA parcellations are needed.
3. Parcellate once per atlas with Connectome Workbench: `wb_command -cifti-parcellate <dtseries> <dlabel> COLUMN <ptseries>`.
4. Behavioural: `unrestricted_*.csv` (NIH Toolbox, NEO-FFI); `restricted_*.csv` for `Family_ID` (Restricted Data application).

`scripts/download_data.py hcp --sample 2` downloads the four dense runs for two subjects (needs boto3 + keys).

Layout: `data/hcp/<subject>/<run>/*.dtseries.nii`, `data/hcp/tables/*.csv`.

## 4. CoRR / NKI — open / registration

CoRR: http://fcon_1000.projects.nitrc.org/indi/CoRR/html/ (S3 `s3://fcp-indi/data/Projects/CoRR/`). NKI-Rockland: http://fcon_1000.projects.nitrc.org/indi/enhanced/ (free registration).

## Secrets

```
export HCP_AWS_ACCESS_KEY_ID=...
export HCP_AWS_SECRET_ACCESS_KEY=...
```
