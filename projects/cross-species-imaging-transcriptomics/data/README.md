# Data acquisition: cross-species-imaging-transcriptomics

Everything is open access. Nothing under `data/` is committed.

## 1. Allen Brain Cell (ABC) Atlas — MERFISH (mouse)

The ABC Atlas is served from a public AWS S3 bucket (`s3://allen-brain-cell-atlas/`, region us-west-2). A JSON manifest per release lists every file with its relative path and size. The documented release used here is `20230830` (later releases add data; check https://alleninstitute.github.io/abc_atlas_access/intro.html for the current list).

Two ways to download:

```bash
# (a) our script: reads the manifest over HTTPS, no AWS account needed
python scripts/download_data.py --sample                 # metadata + gene tables only (~100 MB)
python scripts/download_data.py --merfish --imputed      # full cell-by-gene (h5ad) + imputed values (tens of GB)
python scripts/download_data.py --zhuang                 # Zhuang-ABCA-1..4 metadata + expression

# (b) AWS CLI, anonymous
aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/releases/20230830/
aws s3 sync --no-sign-request s3://allen-brain-cell-atlas/metadata/MERFISH-C57BL6J-638850/20230830/ data/abc/metadata/MERFISH-C57BL6J-638850/20230830/
```

The `abc_atlas_access` package (`pip install abc-atlas-access`) provides `AbcProjectCache` with the same manifest logic; either route produces the layout below.

Key files (directory `MERFISH-C57BL6J-638850`):

| File | Contents |
|---|---|
| `metadata/.../cell_metadata.csv` | one row per cell: `cell_label`, `brain_section_label`, `x`, `y` (section coordinates), `cluster_alias`, donor info |
| `metadata/.../views/cell_metadata_with_cluster_annotation.csv` | adds `class`, `subclass`, `supertype`, `cluster` names and colours |
| `metadata/MERFISH-C57BL6J-638850-CCF/.../ccf_coordinates.csv` | `x_ccf`, `y_ccf`, `z_ccf` (mm) per cell |
| `metadata/MERFISH-C57BL6J-638850-CCF/.../views/cell_metadata_with_parcellation_annotation.csv` | CCFv3 `parcellation_division`, `parcellation_structure`, `parcellation_substructure` per cell |
| `metadata/.../gene.csv` | 500-gene panel (gene symbol, Ensembl id) |
| `expression_matrices/MERFISH-C57BL6J-638850/.../C57BL6J-638850-log2.h5ad` | cell-by-gene, log2(CPM+1) (raw counts also available) |
| `expression_matrices/MERFISH-C57BL6J-638850-imputed/...` | imputed whole-transcriptome values (Yao et al., 2023) |
| `metadata/Allen-CCF-2020/.../parcellation_term.csv` | CCF structure ontology used for the parcellation columns |

Zhuang datasets (`Zhuang-ABCA-1` ... `Zhuang-ABCA-4`) follow the same pattern with a 1,122-gene panel and their own CCF metadata directories.

## 2. Allen Mouse Brain Atlas ISH (mouse, modality comparison)

Structure-level "expression energy" for every ISH experiment via the Allen Brain Map RMA API:

```bash
# list ISH section data sets for a gene symbol (coronal + sagittal)
curl "https://api.brain-map.org/api/v2/data/query.json?criteria=model::SectionDataSet,rma::criteria,[failed\$eq'false'],products[abbreviation\$eq'Mouse'],genes[acronym\$eq'Pvalb'],rma::include,genes,plane_of_section"
# structure unionizes (expression energy per structure) for one SectionDataSet id
curl "https://api.brain-map.org/api/v2/data/query.json?criteria=model::StructureUnionize,rma::criteria,[section_data_set_id\$eq79556706],rma::options[num_rows\$eq'all']"
```

`scripts/download_data.py --ish --genes Pvalb Sst Gad1` wraps these calls with pagination (`start_row`/`num_rows`). The AllenSDK alternative is `allensdk.api.queries.rma_api.RmaApi`. Expression energy is averaged over coronal experiments per gene (sagittal used only when coronal is absent, flagged).

## 3. Allen CCFv3 annotation + ontology

```bash
python -c "from allensdk.core.reference_space_cache import ReferenceSpaceCache as R; R(25, 'annotation/ccf_2017', manifest='data/ccf/manifest.json').get_annotation_volume()"
```
Structure centroids (mm) are computed from the 25-um annotation volume for the 3-D spatial nulls; the ontology CSV is also included in the ABC metadata (`parcellation_term.csv`).

## 4. Allen Human Brain Atlas via abagen (human)

```bash
python -c "import abagen; abagen.fetch_microarray(donors='all', data_dir='data/ahba')"
python -c "import abagen; from abagen import datasets; atlas = abagen.fetch_desikan_killiany(); expr = abagen.get_expression_data(atlas['image'], atlas['info'], data_dir='data/ahba'); expr.to_csv('data/ahba/expression_dk.csv')"
```
Use `lr_mirror='bidirectional'`, `ibf_threshold=0.5`, `probe_selection='diff_stability'`, `norm_matched=True` (abagen defaults / Markello et al., 2021).

## 5. Human MRI maps (neuromaps)

```bash
python -c "from neuromaps import datasets; datasets.fetch_annotation(source='hcps1200', desc='myelinmap'); datasets.fetch_annotation(source='margulies2016', desc='fcgradient01')"
```
Parcellate with `neuromaps.parcellate.Parcellater` onto the same parcellation used for AHBA.

## 6. Mouse MRI maps

- DSURQE atlas + MRI templates (Mouse Imaging Centre, Toronto): https://wiki.mouseimaging.ca/display/MICePub/Mouse+Brain+Atlases (MINC/NIfTI; registered to CCF via ANTs, see the ANTsX mouse mapping tools).
- AMBMC (Australian Mouse Brain Mapping Consortium) T2*-weighted templates: https://imaging.org.au/AMBMC/
- Mouse cortical T1w:T2w map from Fulcher et al. (2019 PNAS): code and data repository referenced in the paper.
- Mouse resting-state fMRI: multi-centre collection of Grandjean et al. (2020 NeuroImage), shared on OpenNeuro; compute a functional gradient with `brainspace` on a CCF-region time-series matrix.

## 7. Orthologs

```bash
python scripts/download_data.py --orthologs   # Ensembl BioMart REST query, one-to-one mouse-human orthologs
```

## Expected layout

```
data/
  abc/
    manifest_20230830.json
    metadata/MERFISH-C57BL6J-638850/20230830/{cell_metadata.csv,gene.csv,views/...}
    metadata/MERFISH-C57BL6J-638850-CCF/20230830/{ccf_coordinates.csv,views/...}
    metadata/Allen-CCF-2020/20230830/parcellation_term.csv
    expression_matrices/MERFISH-C57BL6J-638850/20230830/*.h5ad
    expression_matrices/MERFISH-C57BL6J-638850-imputed/20240831/*.h5ad
    metadata/Zhuang-ABCA-*/...
  ish/
    section_datasets_<gene>.json
    unionize_<section_data_set_id>.json
    ish_expression_energy.csv          # gene x structure, built by the script
  ccf/
    manifest.json, annotation_25.nrrd
  ahba/                                 # abagen cache + expression_<parcellation>.csv
  neuromaps/                            # neuromaps cache
  mouse_mri/                            # DSURQE, AMBMC, T1w:T2w, fMRI gradient
  orthologs/mouse_human_one2one.csv
```
