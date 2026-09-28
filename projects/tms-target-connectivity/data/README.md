# Data acquisition

Nothing in this directory is committed. Record the **access date and snapshot tag**
for every dataset: OpenNeuro datasets are versioned and HCP products have release
identifiers, so an analysis is only reproducible if those are written down.

## Expected directory layout

```
data/
├── hcp/
│   ├── group_avg/
│   │   ├── HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii   # dense FC
│   │   ├── fc_mmp1_group.npy            # parcellated FC, derived
│   │   ├── sc_mmp1_group.npy            # parcellated SC (tractography), derived
│   │   └── release_notes.txt
│   ├── subjects/
│   │   └── <subject_id>/
│   │       ├── MNINonLinear/Results/..._Atlas_MSMAll_hp2000_clean.ptseries.nii
│   │       ├── T1w/T1w_acpc_dc_restore.nii.gz
│   │       └── T1w/Diffusion/
│   └── parcellation/
│       ├── mmp1_labels.txt              # 360 parcel names, matrix order
│       └── mmp1_centroids.txt           # (360, 3) mm coordinates - REQUIRED for nulls
├── openneuro/
│   ├── candidates.json                  # written by scripts/download_data.py
│   ├── ds004024/                        # BIDS tree
│   └── ds005498/
├── headmodels/
│   ├── m2m_<subject>/                   # SimNIBS charm output
│   └── simulations/<subject>/<session>/*.msh
└── derivatives/
    ├── efield_parcellated/<subject>_<session>.npy
    ├── surrogates/<map_name>_variogram_10000.npy
    └── scores/<subject>_<session>_scores.csv
```

## 1. OpenNeuro TMS / tDCS datasets (open, no registration)

Search first — the catalogue changes, so do not rely on a hard-coded list:

```bash
python scripts/download_data.py --search-openneuro --query TMS --query tDCS --query tACS
# -> data/openneuro/candidates.json
```

The query hits the public GraphQL endpoint:

```
POST https://openneuro.org/crn/graphql
{"query": "query($query: JSON!, $first: Int!, $after: String) { datasets: advancedSearch(...) }",
 "variables": {"query": {"query_string": {"query": "TMS"}}, "first": 25}}
```

**Screen every hit by hand.** A keyword match on "TMS" also catches datasets that
merely mention it in a task description or a README. The inclusion criteria that
matter for this project:

- measured, *site-specific* stimulation effects (TMS-evoked EEG potentials or
  concurrent BOLD), not just clinical outcome;
- structural images (T1w, ideally T2w) for each subject, so a head model can be built;
- coil placement documented — neuronavigation transforms, digitised coil position,
  or at minimum a precise scalp landmark. **Without this the field cannot be
  modelled and the dataset is unusable here.**

Then download, restricted by modality (these datasets are large):

```bash
# ds004024 is ~1 TB in full - always restrict
python scripts/download_data.py --download-openneuro ds004024 --modality eeg --modality anat

# or directly
pip install openneuro-py
openneuro-py download --dataset=ds004024 --target-dir=data/openneuro/ds004024 \
    --include='*/eeg/*' --include='*/anat/*'

# or with datalad, which fetches file content on demand
datalad clone https://github.com/OpenNeuroDatasets/ds004024.git data/openneuro/ds004024
datalad get -d data/openneuro/ds004024 'sub-*/ses-*/eeg/*'
```

Known-relevant starting points (verify against the live record):

| Accession | Contents |
|---|---|
| `ds004024` | TMS-EEG + MRI/fMRI/DWI, paired associative stimulation and connectivity, 13 participants, tasks `ccPAS` / `spTMS` / `rest`, BIDS 1.6.0 |
| `ds005498` | Single-pulse concurrent TMS-fMRI |

Further TMS-EEG multimodal datasets have appeared in *Data in Brief* and similar
outlets through 2026; the keyword search is how you find them.

## 2. HCP S1200 (free registration + Open Access Data Use Terms)

```bash
python scripts/download_data.py --hcp-instructions
```

1. Register at https://db.humanconnectome.org, accept the **WU-Minn HCP Open
   Access Data Use Terms**, then generate S3 credentials under Account Settings.
   The **Restricted Data** tier is *not* needed for this project — do not request
   or download restricted variables.

2. Export the credentials; the download script reads them from the environment and
   never writes them anywhere:

   ```bash
   export HCP_AWS_ACCESS_KEY_ID=...
   export HCP_AWS_SECRET_ACCESS_KEY=...
   ```

3. **Group average (the normative arm — start here).** The S1200 Group Average
   release contains MSM-All group-average dense functional connectomes for the
   n = 1003 subjects with complete rfMRI and for the n = 812 r227-recon subset.
   Download it from ConnectomeDB directly (simpler than guessing S3 prefixes):
   https://www.humanconnectome.org/study/hcp-young-adult/article/s1200-group-average-data-release

   A dense connectome is tens of GB. Read it **row-wise only**:

   ```python
   from tms_target.connectome import dense_seed_connectivity
   seed_map = dense_seed_connectivity("data/hcp/group_avg/....dconn.nii", seed_indices=[12345])
   ```

   `nibabel` memory-maps the file, so a few rows are cheap and the whole array is
   not. Never call `np.asarray` on the full dataobj.

4. **Per-subject (the individualized arm — large).** Parcellated timeseries are
   two orders of magnitude smaller than dense ones and are sufficient here:

   ```bash
   aws s3 ls s3://hcp-openaccess/HCP_1200/
   aws s3 cp s3://hcp-openaccess/HCP_1200/<subject>/MNINonLinear/Results/ \
       data/hcp/subjects/<subject>/ --recursive \
       --exclude "*" --include "*Atlas_MSMAll_hp2000_clean.ptseries.nii"
   ```

   Also fetch `T1w/` for head models and `T1w/Diffusion/` for tractography. Start
   with the HCP "100 unrelated subjects" subset rather than all 1200.

5. **Structural connectomes** are not distributed as matrices; generate them with
   MRtrix3 (`5ttgen`, `dwi2response`, `dwi2fod`, `tckgen`, `tck2connectome`) or
   DSI Studio, then `tms_target.connectome.structural_to_weights`.

## 3. Parcellation and centroids

HCP-MMP1 (Glasser et al. 2016, *Nature*) is distributed via BALSA
(https://balsa.wustl.edu). Export, in matrix order:

- `mmp1_labels.txt` — one parcel name per line, 360 lines
- `mmp1_centroids.txt` — 360 × 3 centroid coordinates in mm

**The centroids are not optional.** Both null families need geometry: the spin test
needs positions to rotate, and variogram matching needs a distance matrix. An
analysis without centroids cannot be tested properly.

Schaefer parcellations (200 and 400 parcels) for the sensitivity analysis ship with
`neuromaps` or are available from the ThomasYeoLab GitHub repository.

## 4. E-field models (software, not downloads)

```bash
pip install simnibs          # or the platform installer, which is the supported route
charm <subject_id> <T1.nii.gz> <T2.nii.gz>     # -> m2m_<subject>/
# then script the coil placement from the dataset's neuronavigation record and run
# the simulation; the result is a .msh with 'normE' and 'E' element data
```

```python
from tms_target.efield import load_efield_msh, parcellate_field, normalize_to_motor_threshold
field = load_efield_msh("data/headmodels/simulations/sub-01/ses-1/TMS_scalar.msh",
                        tissue_tags=(2,))     # tag 2 = grey matter
```

For tES, ROAST (https://www.parralab.org/roast/) writes NIfTI; load with
`load_efield_nifti(path, mask_path=gm_mask)` and **always pass a grey-matter mask**,
or scalp voxels will set the map's peak.

The SimNIBS example dataset (`ernie`, MNI152) is the right thing to develop against
before any per-subject model exists:
https://simnibs.github.io/simnibs/build/html/dataset.html

## 5. Surrogate maps: generate once, reuse everywhere

Surrogates are expensive and should be a cached derivative, not recomputed per test.
Reusing one set across comparisons also makes the tests mutually fair.

```python
import numpy as np
from tms_target.nulls import variogram_surrogates

sur = variogram_surrogates(connectivity_map, distances, n_surrogates=10000, seed=0)
np.save("data/derivatives/surrogates/sgc_seed_variogram_10000.npy", sur)
```

## Provenance checklist before analysis

- [ ] OpenNeuro accession numbers **and snapshot tags** recorded
- [ ] HCP release identifier and download date recorded; Open Access terms accepted
- [ ] Coil placement documented for every session, with its uncertainty
- [ ] Parcellation version recorded; centroids present and in matrix order
- [ ] SimNIBS version and conductivity values recorded (they affect absolute V/m)
- [ ] Confirmed no subject overlap between the normative connectome cohort and the
      stimulation datasets
- [ ] Surrogate maps cached with their seed and count in the filename
