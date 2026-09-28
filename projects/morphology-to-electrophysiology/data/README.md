# Data acquisition: morphology-to-electrophysiology

All sources are open. Nothing under `data/` is committed.

## 1. Allen Cell Types Database via allensdk

```bash
pip install allensdk            # brings pynwb/h5py; consider a dedicated conda env (python 3.9-3.11)
cd projects/morphology-to-electrophysiology

# smoke test: 10 mouse + 5 human cells with reconstructions, SWC + feature tables only
python scripts/download_data.py --sample

# full: all cells with reconstructions, both species, SWC files + feature tables
python scripts/download_data.py --species mouse human

# additionally fetch NWB sweep files (large: ~50-300 MB per cell) for own feature extraction
python scripts/download_data.py --species mouse human --nwb
```

What the script does (all via `allensdk.core.cell_types_cache.CellTypesCache`):

- `get_cells(require_reconstruction=True)` → `data/allen/cells.csv` (id, species, layer, area, dendrite_type, transgenic_line, donor_id, ...)
- `get_ephys_features(dataframe=True)` → `data/allen/ephys_features.csv` (input_resistance_mohm, sag, tau, threshold_i_long_square, upstroke_downstroke_ratio_long_square, adaptation, f_i_curve_slope, vrest, latency, avg_isi, ...)
- `get_morphology_features(dataframe=True)` → `data/allen/morphology_features.csv` (Allen's own morphometrics for cross-checking)
- `get_reconstruction(specimen_id)` → `data/allen/specimen_<id>/reconstruction.swc`
- `get_ephys_data(specimen_id)` (with `--nwb`) → `data/allen/specimen_<id>/ephys.nwb`

The Allen SWC convention: coordinates in µm, y axis is the pial-depth axis; type 1 soma, 2 axon, 3 basal, 4 apical.

## 2. Allen Patch-seq (transcriptomic types)

- Portal: https://portal.brain-map.org/explore/classes/multimodal-characterization — download the metadata/feature CSVs (cell id, t-type, MET-type, ephys features, morphology availability) and the reconstruction archives for mouse VISp GABAergic (Gouwens et al. 2020), mouse glutamatergic, and human neocortex (Berg et al. 2021; Chartrand et al. 2023; Lee et al. 2023).
- NWB sweep files are mirrored on the DANDI Archive. Find the Allen Institute Patch-seq dandisets at https://dandiarchive.org (search "Patch-seq" and "Allen Institute"; verify the dandiset ids in the source papers' data-availability statements), then:

```bash
pip install dandi
dandi download DANDI:<dandiset_id> --output-dir data/patchseq/<dandiset_id> --download dandiset.yaml,assets
```

Save the metadata CSVs under `data/patchseq/` and record the exact URLs/dates in `data/patchseq/SOURCES.txt`.

## 3. Allen biophysical models (baseline)

```python
from allensdk.api.queries.biophysical_api import BiophysicalApi
bp = BiophysicalApi()
bp.cache_stimulus = False
bp.cache_data(neuronal_model_id, working_directory='data/biophys/<neuronal_model_id>')
```

Model ids are listed per cell on celltypes.brain-map.org ("Biophysical - perisomatic" / "Biophysical - all active"). Running them requires NEURON (`pip install neuron`) and the Allen model-run scripts inside each downloaded directory (`nrnivmodl modfiles`, then `python -m allensdk.model.biophysical.runner manifest.json`).

## 4. NeuroMorpho.org (morphology-only pre-training set)

```bash
python scripts/download_data.py --neuromorpho --species mouse human --nm-max-pages 4
```

Uses `POST https://neuromorpho.org/api/neuron/select` with `{"species": [...], "brain_region": ["neocortex"]}` and downloads `CNG.swc` files to `data/neuromorpho/<archive>/`. See https://neuromorpho.org/apiReference.html.

## Expected layout

```
data/
  README.md
  allen/
    manifest.json
    cells.csv
    ephys_features.csv
    morphology_features.csv
    specimen_<id>/reconstruction.swc
    specimen_<id>/ephys.nwb              # only with --nwb
  patchseq/
    SOURCES.txt
    <metadata csvs>
    <dandiset_id>/...
  biophys/<neuronal_model_id>/...
  neuromorpho/<archive>/<name>.CNG.swc
```
