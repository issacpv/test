# Data acquisition

Nothing here is committed. IXI and OpenNeuro are open; HCP-YA needs a free registration; HCP-Aging/-Development need an NDA application; Cam-CAN needs a DUA.

## 1. OpenNeuro multi-shell dMRI discovery — open

The OpenNeuro bucket is public. `scripts/download_data.py openneuro-discover` queries the OpenNeuro GraphQL API for datasets with diffusion data, reads each dataset's first `*.bval` file straight from `https://s3.amazonaws.com/openneuro.org/<dsid>/...` and writes `data/openneuro/multishell_candidates.csv` with the shells found. Download a chosen dataset with any of:

```
pip install openneuro-py && openneuro-py download --dataset ds00XXXX --include "sub-*/dwi/*"
# or
aws s3 sync --no-sign-request s3://openneuro.org/ds00XXXX data/openneuro/ds00XXXX --exclude "*" --include "sub-*/dwi/*"
# or
datalad install https://github.com/OpenNeuroDatasets/ds00XXXX.git
```

Layout: `data/openneuro/<dsid>/sub-XX/dwi/sub-XX_dwi.{nii.gz,bval,bvec,json}`.

## 2. HCP Young Adult S1200 — free registration

1. Register at https://db.humanconnectome.org, accept the Open Access terms, enable AWS S3 access and export the keys as `HCP_AWS_ACCESS_KEY_ID` / `HCP_AWS_SECRET_ACCESS_KEY`.
2. Per subject the preprocessed diffusion data live at `s3://hcp-openaccess/HCP_1200/<subject>/T1w/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz,grad_dev.nii.gz}` (~1.3 GB per subject). The 45 retest subjects are in `s3://hcp-openaccess/HCP_Retest/<subject>/T1w/Diffusion/`.
3. `scripts/download_data.py hcp --sample 3 --bvals-only` fetches only the gradient tables (for protocol emulation checks); drop `--bvals-only` for the full volumes.

Layout: `data/hcp/<subject>/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}` and `data/hcp_retest/<subject>/Diffusion/...`.

## 3. HCP Aging / HCP Development — NDA application

Apply through https://nda.nih.gov/ccf (Lifespan Human Connectome Projects, Data Use Certification). Download the "Diffusion preprocessed" packages via the NDA download manager or `nda-tools` (`downloadcmd -dp <package_id>`). Layout: `data/hcp_aging/<subject>/Diffusion/...`, `data/hcp_dev/<subject>/Diffusion/...`.

## 4. Cam-CAN — DUA

Request access at https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/ (data-use agreement). Diffusion data are in the `cc700` release (`dwi/` per subject, b = 1000 and 2000). Layout: `data/camcan/<subject>/dwi/...`.

## 5. IXI — open

The IXI DTI package (15 directions, b = 1000) and its shared `bvecs.txt`/`bvals.txt` are served from https://brain-development.org/ixi-dataset/ (files under `https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/`). `scripts/download_data.py ixi` downloads the gradient tables, and the DTI tarball unless `--tables-only` is given. Demographics are in `IXI.xls` on the same page. Layout: `data/ixi/IXI-DTI/`, `data/ixi/bvals.txt`, `data/ixi/bvecs.txt`, `data/ixi/IXI.xls`.

## 6. Optional: UK Biobank, ADNI-3

UK Biobank (application + fee): NODDI/DTI IDPs are already computed (category 134). ADNI-3 advanced multi-shell dMRI: LONI IDA application; select "Axial DTI - multi-shell" series.

## Secrets

```
export HCP_AWS_ACCESS_KEY_ID=...
export HCP_AWS_SECRET_ACCESS_KEY=...
```
