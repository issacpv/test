# Data acquisition: celltype-composition-mri-contrast

All sources are open. Nothing under `data/` is committed.

## 1. Allen Brain Cell (ABC) Atlas - whole mouse brain MERFISH

The ABC Atlas is served from a public AWS S3 bucket with a `manifest.json` per release; the official
helper package wraps it:

```bash
pip install abc-atlas-access
python scripts/download_data.py --abc --sample        # metadata only, small release subset
python scripts/download_data.py --abc                 # full MERFISH cell metadata (+ CCF coordinates)
```
Programmatically (what the script does):
```python
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
cache = AbcProjectCache.from_s3_cache("data/abc_atlas")
cache.list_directories()                       # e.g. 'MERFISH-C57BL6J-638850', 'MERFISH-C57BL6J-638850-CCF', 'WMB-taxonomy'
cell = cache.get_metadata_dataframe("MERFISH-C57BL6J-638850", "cell_metadata")
ccf  = cache.get_metadata_dataframe("MERFISH-C57BL6J-638850-CCF", "ccf_coordinates")
```
Columns used: `cell_label`, `x/y/z` (reconstructed) or `x_ccf, y_ccf, z_ccf` (mm), `class`, `subclass`,
`cluster`, `parcellation_structure`, `parcellation_division`. Expression matrices (h5ad) are only needed
for the gene-program models (H3): `cache.get_data_path("MERFISH-C57BL6J-638850", "C57BL6J-638850/log2")`.

Without the package, the bucket is `s3://allen-brain-cell-atlas/` (region us-west-2, no credentials);
`aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/releases/` lists releases and their `manifest.json`.

## 2. Allen CCFv3 annotation

```bash
python scripts/download_data.py --ccf --resolution 25
```
downloads `annotation_25.nrrd` (and optionally `average_template_25.nrrd`) from
`http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/`.
The structure ontology (id -> acronym/parent) is fetched from the Allen API
(`http://api.brain-map.org/api/v2/structure_graph_download/1.json`).

## 3. MRI atlases

- **DSURQE** (Mouse Imaging Centre): https://wiki.mouseimaging.ca/display/MICePub/Mouse+Brain+Atlases -
  download the 40 um template (`DSURQE_40micron_average.mnc` / NIfTI) and labels; convert MINC with
  `mnc2nii` if needed. `scripts/download_data.py --dsurqe` fetches from the repository URL listed on that page
  (edit `DSURQE_FILES` in the script if the file names change).
- **AMBMC**: https://imaging.org.au/AMBMC - register on the page, download the T2*-weighted model and labels.
- **Zhang-lab in vivo DTI atlas** (Wu et al., 2013, NeuroImage): distribution named in the paper; store the
  FA/MD/T2 maps under `data/mri/zhang_invivo/`.
- **Blue Brain Cell Atlas** densities (comparison): https://bbp.epfl.ch/nexus/cell-atlas - download the
  neuron/glia density NRRDs.

Register each atlas to CCF with ANTs (`antspyx`) using the DSURQE->CCF or template->CCF transforms; save
the transformed contrasts as `data/mri/<atlas>/<contrast>_ccf25.nii.gz`.

## Expected layout

```
data/
  README.md
  abc_atlas/                      # AbcProjectCache directory (manifest + metadata + optional h5ad)
  ccf/{annotation_25.nrrd, average_template_25.nrrd, structures.csv}
  mri/{dsurqe,ambmc,zhang_invivo}/<contrast>_ccf25.nii.gz
  bbca/*.nrrd
  tables/{densities_by_structure.parquet, contrasts_by_structure.parquet}
```
