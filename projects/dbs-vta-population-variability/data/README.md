# Data acquisition

Everything is open. Only this README is committed.

## 1. NeuroMorpho.org axonal reconstructions (REST API v1)

```bash
python scripts/download_data.py --sample                 # metadata for 3 queries + 10 SWC files
python scripts/download_data.py --metadata --swc         # all queries in QUERIES, all SWC files
python scripts/download_data.py --metadata --swc --species human monkey --regions neocortex
```

API pattern (https://neuromorpho.org/apiReference.html):

- `GET /api/neuron/select?q=species:mouse&q=brain_region:neocortex&page=0&size=500`
  (multiple `q` are ANDed; comma-separated values ORed; `size` <= 500)
- `GET /api/neuron/fields/brain_region` etc. for valid values
- SWC: `https://neuromorpho.org/dableFiles/<archive lower-case>/CNG version/<neuron_name>.CNG.swc`

Records are filtered client-side to those whose `structural_domains` mention the axon
(e.g. "Dendrites, Soma, Axon") and whose `physical_Integrity` is not "Dendrites Complete" only.
Set `NEUROMORPHO_VERIFY_SSL=0` only if the site's TLS chain is temporarily broken.

Output: `data/neuromorpho/meta/<query>.jsonl`, `data/neuromorpho/swc/<neuron_name>.CNG.swc`.

## 2. Janelia MouseLight whole-brain axons

https://ml-neuronbrowser.janelia.org -> select neurons (e.g. soma in MOp/MOs layer 5, axon
targets STN/TH) -> export SWC or JSON. Save under `data/mouselight/*.swc`. Many MouseLight
neurons are also deposited on NeuroMorpho (search archive names with
`GET /api/neuron/fields/archive` and filter by "mouselight"/"Janelia").

## 3. SEU-ALLEN / BICCN full morphologies (Peng et al., 2021)

Deposited to NeuroMorpho and the BICCN portal (https://www.biccn.org, search "full morphology").
Save SWC files under `data/seu_allen/`. Use the soma-region and projection annotations to
select L5 PT neurons with subthalamic collaterals.

## 4. Allen Cell Types (human cortex)

```python
from allensdk.core.cell_types_cache import CellTypesCache
ctc = CellTypesCache(manifest_file="data/allen/manifest.json")
cells = ctc.get_cells(species=["Homo Sapiens"], require_reconstruction=True)
for c in cells:
    ctc.get_reconstruction(c["id"])   # cached as SWC under data/allen/
```

## 5. Lead-DBS electrode models and pathway atlas

```bash
git clone --depth 1 https://github.com/netstim/leaddbs data/leaddbs
ls data/leaddbs/templates/electrode_models/    # *.mat lead geometries (GPL)
```

The lead specifications in `dbs_popvta.lead_field.LEADS` were transcribed from manufacturer
documentation; compare against the `.mat` files (loadable with `scipy.io.loadmat`) if you add a lead.
OSS-DBS/Lead-DBS FEM potentials can be exported as NIfTI and read with `FieldFromNifti`
(requires `nibabel`); save them under `data/fields/`.

## Expected layout

```
data/
  README.md
  neuromorpho/meta/*.jsonl
  neuromorpho/swc/*.CNG.swc
  mouselight/*.swc
  seu_allen/*.swc
  allen/...
  leaddbs/            # git clone (optional)
  fields/*.nii.gz     # optional FEM exports
outputs/
  thresholds_<lead>.parquet
  sobol_indices.csv
  probabilistic_vta_<lead>_<setting>.csv
```
