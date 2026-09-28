# Data acquisition: virtual-mouse-brain-validation

All sources are open. Nothing under `data/` is committed.

## 1. Allen Mouse Brain Connectivity Atlas (allensdk)

```bash
pip install allensdk
python scripts/download_data.py --allen            # writes data/allen/connectome_*.npz + region tables
```
Under the hood (`MouseConnectivityCache`): `get_structure_tree()`, `get_experiments(cre=False)`,
`get_projection_matrix(experiment_ids, structure_ids, hemisphere_ids=[1,2], parameter="normalized_projection_volume")`
and the structure unionizes. The script builds both *normalized projection volume* (density-like) and
*projection energy*-based matrices, ipsi- and contralateral, at the "summary structures" level, and saves the
CCFv3 region centroids used for distance-preserving nulls.

CCFv3 annotation volume (for parcellating fMRI):
`http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd`
(also 50/100 um versions in the same folder).

## 2. Multi-centre mouse resting-state fMRI (OpenNeuro ds001720)

```bash
pip install openneuro-py
openneuro-py download --dataset ds001720 --target-dir data/openneuro/ds001720           # full (~tens of GB)
openneuro-py download --dataset ds001720 --include "sub-0101*" --target-dir data/openneuro/ds001720   # sample
# or: datalad install https://github.com/OpenNeuroDatasets/ds001720.git
# or: aws s3 sync --no-sign-request s3://openneuro.org/ds001720 data/openneuro/ds001720
```
`participants.tsv` carries the site / anaesthesia information used as covariates.
Preprocess with RABIES (https://github.com/CoBrALab/RABIES, container images on Docker Hub):
`rabies preprocess <bids_dir> <out_dir> ...` then `rabies confound_correction ...` to obtain
cleaned time series in CCF space; parcellate with `vmb_validation.parcellate.parcellate_volume`.

## 3. Widefield calcium imaging (DANDI)

Locate the 2025 *Scientific Data* "multimodal dataset linking wide-field calcium imaging to behavior"
dandiset (search https://dandiarchive.org for "wide-field calcium" / "widefield"; the dandiset id is
given in the paper's Data Availability statement), then:

```bash
pip install dandi
dandi ls DANDI:<id>
dandi download DANDI:<id> -o data/dandi/ --existing skip          # or a subset of assets
```
NWB files hold dF/F frames (`ophys` / `OnePhotonSeries`) and the registration to the Allen dorsal map;
`vmb_validation.parcellate.parcellate_frames` turns frames + a label image into region time series.

## 4. MouseLight single-axon projections (alternative connectome)

https://ml-neuronbrowser.janelia.org - export neurons (JSON/SWC with CCF coordinates) and build a
region x region matrix of axonal endpoints per soma region; save as `data/mouselight/connectome.csv`
(row = source region acronym, column = target region acronym). `connectome.load_csv_connectome` reads it.

## Expected layout

```
data/
  README.md
  allen/{connectome_ncd_ipsi.npz, connectome_ncd_contra.npz, regions.csv, centroids.csv, annotation_100.nrrd}
  openneuro/ds001720/...           # BIDS
  rabies/...                       # preprocessed outputs (not committed)
  dandi/<dandiset>/...             # NWB
  mouselight/connectome.csv
  parcellated/{fmri_<site>_<sub>.npz, widefield_<session>.npz}
```
