# Data acquisition: morphology-to-spike-waveform

All sources are open; no registration is needed. Nothing under `data/` is committed.

## 1. Allen Cell Types Database (morphologies, ephys features, models)

### AllenSDK (recommended)

```bash
pip install allensdk          # Python <= 3.11
python scripts/download_data.py --sample                    # 10 mouse cells with reconstructions
python scripts/download_data.py --species mouse --all       # every reconstructed mouse cell + ephys features
python scripts/download_data.py --species human --all
python scripts/download_data.py --models                    # biophysical model files for cells that have them
```

Equivalent calls:

```python
from allensdk.core.cell_types_cache import CellTypesCache, ReporterStatus
ctc = CellTypesCache(manifest_file="data/cell_types/manifest.json")
cells = ctc.get_cells(species=["Mus musculus"], require_reconstruction=True)
morph = ctc.get_reconstruction(cells[0]["id"])          # SWC -> allensdk Morphology object (and .swc on disk)
ephys = ctc.get_ephys_features()                         # per-cell intracellular features (spike width, AHP, ...)
sweeps = ctc.get_ephys_data(cells[0]["id"])              # NWB with raw sweeps
```

### REST API (no AllenSDK)

```bash
# cells with reconstructions (paginate with start_row/num_rows)
curl "https://api.brain-map.org/api/v2/data/query.json?criteria=model::ApiCellTypesSpecimenDetail,rma::criteria,[nr__reconstruction_type\$ne'null'],rma::options[num_rows\$eq'all']"
# SWC for one specimen (well_known_file_type 3DNeuronReconstruction)
curl "https://api.brain-map.org/api/v2/data/query.json?criteria=model::Specimen,rma::criteria,[id\$eq{SPECIMEN_ID}],rma::include,neuron_reconstructions(well_known_files)"
curl -o data/cell_types/swc/{SPECIMEN_ID}.swc "https://api.brain-map.org{download_link}"
```

### Biophysical models

```python
from allensdk.api.queries.biophysical_api import BiophysicalApi
bp = BiophysicalApi()
bp.cache_data(neuronal_model_id, working_directory="data/models/{neuronal_model_id}")   # hoc, mod files, fit json, SWC
```
Model ids come from `model::NeuronalModel,rma::criteria,[specimen_id$eq{ID}]` (template names "Biophysical - all active" or "Biophysical - perisomatic"). Running them needs NEURON (`nrnivmodl modfiles`). The all-active ensembles of Nandi et al. (2022) are available per the paper's data-availability statement (GitHub `AllenInstitute/All-active-Workflow` and associated DANDI/Zenodo records).

## 2. Allen Visual Coding - Neuropixels (opto-tagged in-vivo waveforms)

```python
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest="data/ecephys_cache_dir/manifest.json")
sessions = cache.get_session_table()
opto = sessions[sessions.full_genotype.str.contains("Sst|Pvalb|Vip")]
session = cache.get_session_data(opto.index[0])
wf = session.mean_waveforms[unit_id]                     # xarray channel x time (uV), NP1.0 geometry
probe_channels = session.channels                         # probe_vertical_position, probe_horizontal_position
```
`python scripts/download_data.py --opto-sessions 2` caches the first two opto-tagging sessions. Opto-tag identification follows the AllenSDK optotagging tutorial (evoked-rate ratio in the 10-ms window after the light pulse).

## 3. Neuropixels Ultra opto-tagging (Ye et al., 2025 Neuron)

Data are on the DANDI archive per the paper's data-availability statement:

```bash
pip install dandi
dandi download DANDI:<dandiset id from the paper> -o data/npultra/
```

## 4. NeuroMorpho.org (additional mouse V1 morphologies)

```bash
curl -X POST "https://neuromorpho.org/api/neuron/select?page=0&size=100" -H "Content-Type: application/json" \
     -d '{"species":["mouse"],"brain_region":["neocortex"],"cell_type":["pyramidal"]}'
# SWC: https://neuromorpho.org/dableFiles/{archive}/CNG%20version/{neuron_name}.CNG.swc
```

## Expected layout

```
data/
  cell_types/
    manifest.json
    cells.csv                        # metadata (species, dendrite_type, layer, Cre line, has_model)
    ephys_features.csv
    specimen_{id}/reconstruction.swc # written by CellTypesCache
  models/{neuronal_model_id}/        # hoc, modfiles/, fit_parameters.json, *.swc
  ecephys_cache_dir/                 # AllenSDK Neuropixels cache (opto sessions)
  npultra/                           # DANDI download
  neuromorpho/*.CNG.swc
  simulations/                       # outputs of the factorial sweeps (never committed)
```
