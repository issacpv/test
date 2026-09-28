# Data acquisition

Everything below is open. Nothing in `data/` is committed. Expected layout at the end:

```
data/
  abc_atlas/                      # ABC Atlas cache (abc_atlas_access layout, mirrors S3 keys)
    releases/20241115/manifest.json
    metadata/WMB-10X/20241115/cell_metadata_with_cluster_annotation.csv
    metadata/WMB-taxonomy/20231215/cluster_to_cluster_annotation_membership.csv
    metadata/MERFISH-C57BL6J-638850/20241115/cell_metadata_with_cluster_annotation.csv
    metadata/MERFISH-C57BL6J-638850-CCF/20231215/cell_metadata_with_parcellation_annotation.csv
    metadata/WHB-10Xv3/20241115/cell_metadata.csv
    metadata/WHB-taxonomy/20240330/cluster_annotation_term.csv
    expression_matrices/WMB-10Xv3/20230630/WMB-10Xv3-TH-log2.h5ad      # etc, one per dissection
    expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad
    expression_matrices/WHB-10Xv3/20240330/WHB-10Xv3-Neurons-log2.h5ad
  cellxgene/siletti/*.h5ad        # optional alternative for the human atlas
  gwas/
    big40/IDPs.csv                # IDP index table
    big40/0001.txt.gz ...         # one file per IDP (discovery stats)
    pgc/*.gz                      # PGC downloads (manual click-through)
    gwas_catalog/GCST*/           # GWAS Catalog harmonised stats
  reference/
    magma/NCBI38.gene.loc, g1000_eur.{bed,bim,fam}
    ldsc/1000G_EUR_Phase3_baseline/, weights_hm3_no_hla/, baselineLD_v2.2/
    orthologs/mouse_human_one2one.tsv
```

## 1. Allen Brain Cell Atlas (open, S3, no credentials)

Option A – the official cache (recommended):

```bash
pip install abc_atlas_access
python - <<'EOF'
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
cache = AbcProjectCache.from_s3_cache(Path("data/abc_atlas"))
print(cache.current_manifest)                    # e.g. releases/20241115/manifest.json
print(cache.list_directories)                    # WMB-10Xv3, MERFISH-C57BL6J-638850, WHB-10Xv3, ...
cache.get_metadata_path("WMB-10X", "cell_metadata_with_cluster_annotation")
cache.get_metadata_path("MERFISH-C57BL6J-638850-CCF", "cell_metadata_with_parcellation_annotation")
cache.get_data_path("MERFISH-C57BL6J-638850", "C57BL6J-638850-log2")
EOF
```

Option B – our manifest-driven downloader (no extra dependency, resumable, size-capped):

```bash
python scripts/download_data.py abc --sample            # metadata + taxonomy only (~2 GB)
python scripts/download_data.py abc --merfish           # + MERFISH log2 matrix (7.6 GB)
python scripts/download_data.py abc --wmb-10x TH HY     # selected 10Xv3 dissections
python scripts/download_data.py abc --human             # WHB-10Xv3 neurons + non-neurons (~40 GB)
```

Option C – AWS CLI: `aws s3 sync --no-sign-request s3://allen-brain-cell-atlas/metadata/ data/abc_atlas/metadata/`.

Manifest URL: `https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/releases/20241115/manifest.json`.
Key metadata columns (MERFISH-CCF file): `cell_label, cluster_alias, class, subclass, supertype, cluster,
x_ccf, y_ccf, z_ccf, parcellation_division, parcellation_structure, parcellation_substructure`.

## 2. Siletti et al. 2023 human atlas via CELLxGENE (alternative to WHB-10Xv3)

```bash
pip install cellxgene-census        # or use the curation REST API used by the downloader
python scripts/download_data.py siletti --list
python scripts/download_data.py siletti --dataset <dataset_id>
```
The downloader queries `https://api.cellxgene.cziscience.com/curation/v1/collections/<collection_id>`
and follows the `assets[].url` links for `.h5ad`. The collection is titled
"Transcriptomic diversity of cell types across the adult human brain"; find its id on cellxgene.cziscience.com
if the default in the script has rotated.

## 3. Allen Human Brain Atlas (microarray) via abagen

```bash
pip install abagen
python -c "import abagen; abagen.fetch_microarray(donors='all', data_dir='data/ahba')"
```

## 4. GWAS summary statistics

### BIG40 (UK Biobank imaging GWAS; open)
- Index: https://open.oxcin.ox.ac.uk/ukbiobank/big40/ → download `IDPs.csv` (IDP id, name, category, h²).
- Per-IDP files: `https://open.oxcin.ox.ac.uk/ukbiobank/big40/release2/stats33k/<IDP>.txt.gz`
  (IDP zero-padded to 4 digits). Columns: `chr rsid pos a1 a2 beta se pval(-log10)`. Check the site
  for the current release path; `scripts/download_data.py big40 --idps 0001 0002 ...` takes a `--base-url`.
- Discovery (`stats22k`) and replication (`stats11k`) files exist alongside the combined `stats33k`.

### PGC (psychiatric; open after click-through)
https://pgc.unc.edu/for-researchers/download-results/ — download manually into `data/gwas/pgc/`
(SCZ3 `PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv.gz`, BD `pgc-bip2021-all.vcf.tsv.gz`,
MDD, ADHD2022, ASD2019, PTSD, AN). The script prints the list of files it expects.

### GWAS Catalog (open, REST + FTP)
```bash
python scripts/download_data.py gwas-catalog --accessions GCST90027158 GCST009325
```
FTP path is computed from the accession number: `.../summary_statistics/GCST90027001-GCST90028000/GCST90027158/`.
Use `harmonised/` sub-folders when present.

## 5. Reference files

- MAGMA: https://cncr.nl/research/magma/ → binary v1.10, `NCBI38.gene.loc` (or NCBI37 to match GWAS build),
  `g1000_eur` reference panel.
- LDSC: `git clone https://github.com/bulik/ldsc`; baseline-LD v2.2, HapMap3 weights and 1000G EUR
  Phase 3 plink files from the LDSC "Partitioned heritability" wiki.
- Orthologs: Ensembl BioMart, filter `Mouse orthology confidence = 1` and one-to-one; save as
  `data/reference/orthologs/mouse_human_one2one.tsv` with columns `mouse_symbol, human_symbol, human_ensembl`.
