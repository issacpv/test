# Data acquisition: gene-gradients-neural-timescales

All sources are open access; no registration is needed. Nothing under `data/` is committed.

## 1. Allen Visual Coding - Neuropixels (AllenSDK)

The AllenSDK `EcephysProjectCache` downloads session NWB files from the public S3 bucket `allen-brain-observatory` (also browsable at https://registry.opendata.aws/allen-brain-observatory/).

```bash
pip install allensdk            # Python <= 3.11
python scripts/download_data.py --sample          # session table + 1 session (~2 GB) + unit table
python scripts/download_data.py --visual-coding   # all 58 sessions (~1 TB)
```

What the script does (equivalent AllenSDK calls):

```python
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest="data/ecephys_cache_dir/manifest.json")
sessions = cache.get_session_table()                 # 58 sessions, genotype, areas
units = cache.get_units()                            # all units with QC metrics + CCF coords
session = cache.get_session_data(session_id)         # NWB: spike_times, stimulus_presentations, csd
```

Direct S3 (no AllenSDK): `aws s3 ls --no-sign-request s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/`.

Layer assignments: `session.get_current_source_density(probe_id)` gives the CSD; L4 sink depth is used to assign layers to units by `probe_vertical_position` (see Siegle et al., 2021 methods). Opto-tagging: sessions from Sst-IRES-Cre, Pvalb-IRES-Cre and Vip-IRES-Cre mice have `optogenetic_stimulation_epochs`.

## 2. Allen Visual Behavior - Neuropixels

```python
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir="data/vbn_cache")
sessions = cache.get_ecephys_session_table()
```
`python scripts/download_data.py --visual-behavior --n-sessions 5`

## 3. IBL brain-wide map (ONE API)

```bash
pip install ONE-api ibllib
python -c "from one.api import ONE; ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)"
python scripts/download_data.py --ibl --n-insertions 5
```
The public Alyx server needs no login for the brain-wide-map release (user `intbrainlab`, password `international` are the documented public credentials, read from `IBL_USER`/`IBL_PASS` if set). Spike sorting outputs are loaded with `brainbox.io.one.SpikeSortingLoader(pid=..., one=one)`; `clusters['atlas_id']` gives the CCF area.

## 4. Allen Mouse Brain Atlas ISH (expression energy)

```bash
python scripts/download_data.py --ish --genes Hcn1 Kcna1 Scn1a Grin2b Pvalb Sst
python scripts/download_data.py --ish --gene-list data/gene_lists/channel_genes.txt
```
Uses the RMA API (`https://api.brain-map.org/api/v2/data/query.json`), `SectionDataSet` filtered by gene acronym and `StructureUnionize` per data set with `start_row`/`num_rows` pagination. Output: `data/ish/ish_expression_energy.csv` (gene, section_data_set_id, plane, structure_id, expression_energy).

## 5. ABC Atlas MERFISH (cell-level expression, subclass, layer)

```bash
python scripts/download_data.py --merfish-metadata     # cell metadata, parcellation, gene table
aws s3 sync --no-sign-request s3://allen-brain-cell-atlas/expression_matrices/MERFISH-C57BL6J-638850/20230830/ data/abc/expression_matrices/MERFISH-C57BL6J-638850/20230830/
```
Files: `cell_metadata_with_cluster_annotation.csv`, `cell_metadata_with_parcellation_annotation.csv` (CCFv3 `parcellation_structure`, `parcellation_substructure` which includes cortical layers, e.g. `VISp2/3`), `C57BL6J-638850-log2.h5ad`. See https://alleninstitute.github.io/abc_atlas_access/.

## 6. Hierarchy scores and CCF

- Harris et al. (2019 Nature) hierarchy scores: supplementary tables of the paper (cortical and thalamic areas); save as `data/hierarchy/harris2019_hierarchy.csv` with columns `area, hierarchy_score`.
- CCFv3 structure centroids: `allensdk.core.reference_space_cache.ReferenceSpaceCache(25, 'annotation/ccf_2017')`; the script writes `data/ccf/structure_centroids_mm.csv`.

## Expected layout

```
data/
  ecephys_cache_dir/           # AllenSDK Visual Coding cache (manifest.json, sessions/*.nwb, units.csv)
  vbn_cache/                   # Visual Behavior Neuropixels cache
  ibl/                         # ONE cache (spikes, clusters, channels per insertion)
  ish/ish_expression_energy.csv
  abc/metadata/..., abc/expression_matrices/...
  hierarchy/harris2019_hierarchy.csv
  ccf/structure_centroids_mm.csv
  gene_lists/*.txt             # candidate families (HCN, KCN*, SCN*, GRIN*, GABR*)
```
