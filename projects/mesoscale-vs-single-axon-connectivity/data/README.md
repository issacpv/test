# Data acquisition: mesoscale-vs-single-axon-connectivity

All sources are open. Nothing under `data/` is committed.

## 1. Allen Mouse Brain Connectivity Atlas (allensdk)

```bash
pip install allensdk
cd projects/mesoscale-vs-single-axon-connectivity

# smoke test: structure tree + 25 um annotation + 10 wild-type experiments -> small projection matrix
python scripts/download_data.py --allen --sample

# full: all wild-type experiments, summary-structure projection matrix (ipsi/contra), both parameters
python scripts/download_data.py --allen
# add Cre-line experiments as well
python scripts/download_data.py --allen --cre
```

What is fetched via `allensdk.core.mouse_connectivity_cache.MouseConnectivityCache(resolution=25)`:

- `get_structure_tree()` → ontology (structure graph id 1); summary structures = structure set id 167587189 ("Mouse Connectivity - Summary").
- `get_annotation_volume()` → `data/allen/annotation_25.nrrd` (CCFv3 2017 labels, axes = AP, DV, LR; 25 µm voxels; array shape 528 × 320 × 456).
- `get_experiments(cre=False)` → wild-type anterograde injections (metadata incl. `injection_structures`, `primary_injection_structure`, `injection_x/y/z`).
- `get_structure_unionizes(...)` / `get_projection_matrix(...)` → `data/allen/projection_matrix_<parameter>.csv` (rows = experiments, columns = `<acronym>_<ipsi|contra>`), plus `experiments.csv`.

Hemispheres: Allen injections are in the right hemisphere; `hemisphere_id` 1 = left (contralateral to injection), 2 = right (ipsilateral), 3 = both.

Voxel model (optional): `pip install git+https://github.com/AllenInstitute/mouse_connectivity_models` then `meso_vs_axon.allen_connectivity.voxel_model_row(soma_xyz_um, cache_dir)`.

## 2. MouseLight (Janelia) single neurons

- Neuron browser: https://ml-neuronbrowser.janelia.org — select neurons (or all) and export as **JSON** or **SWC**. Coordinates are CCFv3 in µm; JSON nodes carry `allenId` (CCF structure id), which the pipeline uses as a consistency check.
- Save exports under `data/mouselight/` (e.g. `data/mouselight/mouselight_all.json`). Then:

```bash
python scripts/download_data.py --mouselight-json data/mouselight/mouselight_all.json
```

which converts every neuron to `data/swc/mouselight/<idString>.swc` (CCF µm) with a metadata CSV.

- MouseLight neurons are also on NeuroMorpho.org. NOTE: NeuroMorpho's standardised `CNG.swc` files are translated (soma at origin) and therefore **lose CCF registration**; use the browser exports above, or NeuroMorpho's *Source-Version* files (`https://neuromorpho.org/dableFiles/<archive>/Source-Version/<neuron_name>.<ext>`, extension per record's `original_format`).

## 3. SEU-ALLEN full morphologies (Peng et al. 2021; BICCN)

- Brain Image Library (BIL): follow the data-availability statement of Peng et al. (2021, Nature) to the BIL collection of the 1,741 CCF-registered reconstructions (and later BICCN releases with thousands more). BIL supports HTTPS/Globus downloads; place SWC files under `data/swc/seu_allen/`.
- NeuroMorpho.org mirror: query by the paper's PubMed id:

```bash
python scripts/download_data.py --neuromorpho --pmid 34616072 --source-version
```

`--source-version` fetches original-coordinate files where available (see the note above); without it you get CNG files (fine for morphometrics, not for CCF assignment).

## 4. Other single-neuron projectomes

- ION prefrontal cortex (Gao et al. 2022, Nat Neurosci) and hippocampus (Qiu et al. 2024, Science): follow the papers' data portals; save as `data/swc/ion/`. Check the coordinate convention (some releases are in 10 µm CCF voxel units; convert to µm).
- MAPseq/BARseq tables (Han et al. 2018, Nature): supplementary tables; save under `data/mapseq/`.

## Expected layout

```
data/
  README.md
  allen/
    manifest.json
    annotation_25.nrrd
    structures_summary.csv
    experiments.csv
    projection_matrix_normalized_projection_volume.csv
    projection_matrix_projection_density.csv
  mouselight/
    mouselight_all.json
  swc/
    mouselight/<idString>.swc  + metadata.csv
    seu_allen/<name>.swc       + metadata.csv
    ion/...
  mapseq/
```
