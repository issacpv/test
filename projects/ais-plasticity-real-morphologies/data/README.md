# Data acquisition: ais-plasticity-real-morphologies

All sources are open. Nothing under `data/` is committed.

## 1. NeuroMorpho.org somatodendritic reconstructions

REST API v1 at `https://neuromorpho.org/api` (reference https://neuromorpho.org/apiReference.html).
`POST /neuron/select?page=P&size=S` with a JSON body of field -> list of values, e.g.
`{"species": ["mouse"], "cell_type": ["pyramidal"], "brain_region": ["neocortex"]}`; responses are
HATEOAS pages with `_embedded.neuronResources` and `page.totalPages`. SWC files live at
`https://neuromorpho.org/dableFiles/<archive_lowercase>/CNG version/<neuron_name>.CNG.swc`.

```bash
cd projects/ais-plasticity-real-morphologies
pip install -r requirements.txt

# smoke test: 20 mouse pyramidal records, 5 SWC files
python scripts/download_data.py --sample

# full harvest of the five target classes for mouse, rat and human
python scripts/download_data.py --species mouse rat human --classes pyramidal granule interneuron --swc --workers 4
```

Keep the metadata fields `archive`, `reconstruction_software`, `shrinkage_corrected`, `slicing_thickness`,
`physical_Integrity`, `structural_domains`, `brain_region`, `cell_type`, `min_age`, `reference_pmid`.
Records are filtered locally to "Dendrites Complete" integrity (the axon may be missing).
If NeuroMorpho's TLS chain is broken on the day, set `NEUROMORPHO_VERIFY_SSL=0` for the session only.

## 2. ModelDB channel models

Search https://modeldb.science for the following papers and download the model archives (each contains
`.mod` channel mechanisms and hoc/Python templates):

- Hu, Tian, Zhang, ... Shu (2009) *Nature Neuroscience* - Nav1.2 / Nav1.6 AIS distribution model.
- Hallermann, de Kock, Stuart & Kole (2012) *Nature Neuroscience* - AIS Na channel model, L5 pyramidal.
- Gulledge & Bravo (2016) *eNeuro* - simplified and reconstructed neurons with variable AIS.
- Goethals & Brette (2020) *eLife* - resistive-coupling models (code also on the authors' repositories).

Place each archive under `data/modeldb/<accession_or_name>/` and run `nrnivmodl` inside the folder
holding the `.mod` files before using `ais_plasticity.neuron_builder`.

## 3. Allen Cell Types (matched morphology + electrophysiology)

```bash
pip install allensdk
python - <<'EOF'
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
ctc = CellTypesCache(manifest_file='data/allen/manifest.json')
cells = ctc.get_cells(require_reconstruction=True)
feats = ctc.get_ephys_features()          # rheobase, threshold_v, peak_v, upstroke, ...
for c in cells:
    ctc.get_reconstruction(c['id'])       # SWC files cached under data/allen/
EOF
```

## 4. Empirical AIS geometry tables

From the supplementary material of Hamada et al. (2016, PNAS) and the 2025 *Cerebral Cortex* hippocampal
AIS population study, build `data/empirical_ais.csv` with columns
`source, species, region, cell_class, ais_distance_um, ais_length_um, dendrite_size_metric, dendrite_size_value`.

## Expected layout

```
data/
  README.md
  metadata/neurons_<species>.jsonl, neurons.parquet
  swc/<archive>/<neuron_name>.CNG.swc
  modeldb/<entry>/...
  allen/...
  empirical_ais.csv
outputs/   (simulation results, git-ignored)
```
