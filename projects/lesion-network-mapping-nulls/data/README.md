# Data acquisition — lesion-network-mapping-nulls

All data live under `data/` (git-ignored). Credentials are read from environment variables only.

## 1. ATLAS v2.0 lesion masks (INDI; free DUA; encrypted tarball)

1. Read and sign the Data Use Agreement linked from https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html (requires a NITRC account). The download instructions and the decryption password are e-mailed after approval.
2. Follow the e-mailed instructions; the release is an openssl-encrypted tarball hosted on the INDI S3 bucket. Typical steps (adapt to the e-mail):

```bash
export ATLAS_PASSWORD='...'            # from the DUA e-mail; never commit
aws s3 cp --no-sign-request s3://fcp-indi/data/Projects/ATLAS/ATLAS_R2.0_encrypted.tar.gz data/atlas/
openssl aes-256-cbc -md sha256 -d -a -in data/atlas/ATLAS_R2.0_encrypted.tar.gz -out data/atlas/ATLAS_R2.0.tar.gz -pass env:ATLAS_PASSWORD
tar -xzf data/atlas/ATLAS_R2.0.tar.gz -C data/atlas/
```

`python scripts/download_data.py --source atlas` prints these steps and checks the env var; it does not bypass the DUA.

Expected layout (BIDS-like, as released):

```
data/atlas/ATLAS_2/Training/R001/sub-r001s001/ses-1/anat/
    sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
    sub-r001s001_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
data/atlas/ATLAS_2/Training/*/...  (655 subjects with masks; Testing/ has no masks)
data/atlas/ATLAS_2/20220425_ATLAS_2.0_MetaData.csv
```

## 2. Aphasia Recovery Cohort (ARC; OpenNeuro, open)

Search OpenNeuro for "Aphasia Recovery Cohort" (accession ds004884 at time of writing) and download:

```bash
python scripts/download_data.py --source openneuro --dataset ds004884 --sample          # 2 subjects: anat + lesion masks + participants.tsv
python scripts/download_data.py --source openneuro --dataset ds004884 --include 'participants|lesion|T1w|T2w'
# alternatives: pip install openneuro-py && openneuro-py download --dataset ds004884
#               aws s3 sync --no-sign-request s3://openneuro.org/ds004884 data/openneuro/ds004884
```

Behavioural scores (WAB-R AQ, naming, fluency, comprehension, ...) are in `participants.tsv` / `phenotype/`. Lesion masks are in each subject's `anat/` folder (`*_desc-lesion_mask.nii.gz`, native T2 space; register to MNI with ANTs).

## 3. SOOP acute stroke cohort (OpenNeuro ds004889; imaging open, behaviour on request)

```bash
python scripts/download_data.py --source openneuro --dataset ds004889 --sample
```

## 4. Normative connectomes

- HCP-YA 100 unrelated subjects (free registration + open-access terms at https://db.humanconnectome.org/). With AWS credentials issued by ConnectomeDB:

```bash
export AWS_ACCESS_KEY_ID=... ; export AWS_SECRET_ACCESS_KEY=...
aws s3 cp s3://hcp-openaccess/HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz data/hcp/100307/
```

The list of the "100 unrelated subjects" is in the HCP S1200 release notes / ConnectomeDB subject group.

- GSP 1000 (optional): register on Harvard Dataverse (https://dataverse.harvard.edu/dataverse/GSP) and download the preprocessed rfMRI.
- HCP-842 tractography template (open): https://brain.labsolver.org/hcp_template.html (`HCP1065` / `HCP842` `.fib.gz` files for DSI Studio). Parcel-level streamline counts are exported with DSI Studio's `--action=ana --connectivity=...`.
- Arterial territory atlas (Liu et al., 2023, Scientific Data): search NITRC for "arterial territories atlas"; place the MNI NIfTI at `data/atlases/arterial_territories_MNI.nii.gz`.
- Schaefer-400/1000 and Tian S2 parcellations: `nilearn.datasets.fetch_atlas_schaefer_2018`; https://github.com/yetianmed/subcortex.

## Derived products

```
outputs/
  connectome/fc_voxel_parcel_schaefer400.npy      # (n_gm_voxels, 400) Fisher-z
  connectome/sc_parcel_parcel_hcp842.npy
  lesions/<cohort>_features.tsv                   # volume, hemisphere, territory fractions
  maps/<cohort>_<symptom>_<method>.tsv
  nulls/<cohort>_<family>_bias_atlas.nii.gz
  simulations/*.parquet
```
