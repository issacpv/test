# Data acquisition

## 1. NeuroMorpho.org (open, REST API v1, no registration)

Endpoints (https://neuromorpho.org/apiReference.html):

- `GET  https://neuromorpho.org/api/neuron/fields/reconstruction_software` - distinct software labels
- `POST https://neuromorpho.org/api/neuron/select?page=P&size=500` with JSON body
  `{"reconstruction_software": ["Neurolucida"], "species": ["mouse"]}` - filtered, paginated metadata
- `GET  https://neuromorpho.org/dableFiles/<archive>/CNG%20version/<neuron_name>.CNG.swc` - standardised SWC

The downloader stratifies by software x archive and caps the number of cells per archive:

```bash
python scripts/download_data.py --sample                       # a few software classes, 20 records each, 5 SWC each
python scripts/download_data.py --software-list                # print software labels with counts
python scripts/download_data.py --software Neurolucida neuTube Vaa3D "Imaris" --cap-per-archive 300 --swc
python scripts/download_data.py --all-software --cap-per-archive 200 --swc --workers 4
```

Output: `data/metadata/neurons_<software>.jsonl`, `data/metadata/neurons.parquet` (flattened),
`data/swc/<archive>/<neuron_name>.CNG.swc`.

If the NeuroMorpho TLS chain is temporarily broken, set `NEUROMORPHO_VERIFY_SSL=0` for that run only.

## 2. BigNeuron gold-standard set (open)

Manubens-Gil et al., 2023, Nature Methods. Follow the paper's Data Availability statement for the
image stacks, manual gold-standard SWCs and the per-algorithm reconstructions. Place them as:

```
data/bigneuron/<dataset>/<image_id>/gold.swc
data/bigneuron/<dataset>/<image_id>/<algorithm_name>.swc
```

The feature extractor treats `algorithm_name` as the provenance label and `image_id` as the group.

## 3. DIADEM challenge (open)

http://diademchallenge.org - download the gold-standard SWC sets for the six challenge datasets into
`data/diadem/<dataset>/`.

## 4. Allen Cell Types Database (open, `allensdk`)

```python
from allensdk.core.cell_types_cache import CellTypesCache
ctc = CellTypesCache(manifest_file="data/allen/manifest.json")
cells = ctc.get_cells(require_reconstruction=True)
for c in cells:
    ctc.get_reconstruction(c["id"])   # cached as SWC under data/allen/
```

## 5. MouseLight / whole-brain fMOST (open)

MouseLight neurons: http://ml-neuronbrowser.janelia.org (export SWC/JSON). fMOST single-neuron
reconstructions (Peng et al., 2021, Nature) are distributed via the Brain Image Library; place SWCs
under `data/single_pipeline/<source>/`.

## Expected layout

```
data/
  README.md
  metadata/
    software_counts.json
    neurons_<software>.jsonl
    neurons.parquet
  swc/<archive>/<neuron_name>.CNG.swc
  bigneuron/...
  diadem/...
  allen/...
  single_pipeline/...
  features/
    neuromorpho_features.parquet     (one row per SWC: morphometrics + sampling fingerprint + labels)
    bigneuron_features.parquet
```

Nothing under `data/` is committed.
