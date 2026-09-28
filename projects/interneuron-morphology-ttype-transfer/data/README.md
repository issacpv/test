# Data acquisition: interneuron-morphology-ttype-transfer

All sources are open. Nothing under `data/` is committed.

## 1. Allen Institute Patch-seq (mouse V1 GABAergic; human MTG interneurons)

Entry point: https://portal.brain-map.org/explore/classes/multimodal-characterization
(the "Multimodal characterization of individual neurons" pages list the mouse V1 GABAergic
Patch-seq release of Gouwens et al. 2020 and the human interneuron release of Lee et al. 2023,
each with a metadata CSV, morphology SWC archive, and links to NWB electrophysiology on DANDI).

Steps:
1. From the release page download the *specimen metadata* CSV (columns include specimen id,
   transcriptomic type / `t-type`, mapping confidence, MET-type where available, layer, soma depth,
   morphology availability) and save it as `data/allen/mouse_v1_metadata.csv` or
   `data/allen/human_mtg_metadata.csv`.
2. Download the morphology archive linked on the same page (SWC files, one per specimen) and unpack
   into `data/allen/swc/mouse_v1/` and `data/allen/swc/human_mtg/`.
3. Alternatively, per-specimen SWC files can be fetched with allensdk for cells in the Cell Types
   Database (`CellTypesCache.get_reconstruction(specimen_id)`).

`scripts/download_data.py --allen` fetches the Cell Types Database reconstructions and metadata
through allensdk for the subset that is exposed there, and prints the portal links for the rest.

Electrophysiology (not needed for this project's primary analysis) is on DANDI; use
`dandi ls DANDI:<id>` / `dandi download` with the dandiset ids given on the portal pages.

## 2. Tolias-lab mouse M1 Patch-seq mini-atlas (Scala et al., 2021)

```bash
git clone https://github.com/berenslab/mini-atlas data/mini-atlas
```
The repository documents where the SWC morphologies, t-type labels and soma depths are (some files
are hosted on Zenodo and linked from the README). Record the file paths you used in
`data/mini-atlas/PATHS.txt`.

## 3. Taxonomies

- Human MTG snRNA-seq taxonomy (Hodge et al., 2019): https://portal.brain-map.org/atlases-and-data/rnaseq
  (download the cluster annotation table; needed only for the finer t-type level).
- Cross-species consensus subclass mapping: Bakken et al. (2021, Nature) supplementary tables.
Save as `data/taxonomy/*.csv`. `morph_ttype.taxonomy` already contains the marker-token rules for
the five harmonised subclasses; use these tables to audit them.

## 4. Laturnus et al. (2020) benchmark

https://zenodo.org/records/3716519 - reference representations and evaluation code for mouse V1
interneuron morphologies; unpack into `data/laturnus2020/`.

## Unified table

After the downloads, build `data/cells.parquet` with columns:
`cell_id, dataset (allen_v1 | allen_mtg | tolias_m1), species, region, layer, soma_depth_norm,
t_type, mapping_confidence, subclass_harmonised, swc_path, has_axon`.
`scripts/download_data.py --build-table` assembles it from the files above.

## Expected layout

```
data/
  README.md
  allen/{mouse_v1_metadata.csv, human_mtg_metadata.csv, swc/mouse_v1/*.swc, swc/human_mtg/*.swc, manifest.json}
  mini-atlas/...
  taxonomy/*.csv
  laturnus2020/...
  cells.parquet
```
