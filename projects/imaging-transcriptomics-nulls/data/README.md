# Data acquisition: imaging-transcriptomics-nulls

Expected layout:

```
data/
  abagen/                          # AHBA microarray cache (abagen data_dir), ~4 GB for 6 donors
  expression/
    glasser360_expression.parquet  # regions x genes (abagen defaults), donor-averaged
    glasser360_donors/             # per-donor tables for differential stability
    schaefer400_expression.parquet
  parcellations/
    Glasser_MMP1.0.{L,R}.32k_fs_LR.label.gii   # BALSA / hcp_utils
    Schaefer2018_400Parcels_7Networks.{L,R}.32k_fs_LR.label.gii  # netneurotools
    S1200.{L,R}.sphere.32k_fs_LR.surf.gii       # sphere for spin tests (HCP S1200 group avg)
    centroids_glasser.csv  centroids_schaefer400.csv
    geodesic_glasser.npy   geodesic_schaefer400.npy
  maps/
    hcps1200_myelinmap_glasser.csv     # T1w/T2w (Glasser & Van Essen, 2011)
    hcps1200_thickness_glasser.csv
    margulies2016_fcgradient01_glasser.csv
    hcp_aging_age_effect_thickness_glasser.csv   # computed from HCP-A (NDA), see 4.
    ... one CSV per benchmark map (region,value)
  genesets/
    go-basic.obo, gene2go.gz            # Gene Ontology (open)
    hse_genes.txt, brain_specific_genes.txt, ...  # literature gene lists (typed from papers)
```

## 1. Allen Human Brain Atlas via abagen (open, ~4 GB)

```bash
pip install abagen
python scripts/download_data.py ahba --sample          # one donor (9861), ~700 MB, for smoke tests
python scripts/download_data.py ahba                   # all six donors
python scripts/download_data.py expression --atlas glasser   # parcellate (needs parcellation + atlas_info)
```

`abagen.fetch_microarray(donors='all', data_dir='data/abagen')` downloads
from the Allen API. Parcellation uses `abagen.get_expression_data` with the
defaults in `src/imgtx_nulls/expression.py` (probe selection by differential
stability, bidirectional mirroring, `missing='interpolate'`, SRS
normalisation). Two donors have right-hemisphere samples; for benchmark
fidelity run both "left hemisphere only" and "mirrored" variants.

## 2. Parcellations and sphere surfaces (open / free registration)

- **Schaefer 2018** (100-1000 parcels, fsLR 32k): `netneurotools.datasets.fetch_schaefer2018('fslr32k')`
  or `nilearn.datasets.fetch_atlas_schaefer_2018` (MNI volume, for abagen volumetric mode).
- **Glasser MMP1.0** (360 parcels): distributed via BALSA (https://balsa.wustl.edu,
  free registration; file `Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii`).
  Convert to per-hemisphere label GIFTIs with `wb_command -cifti-separate ... -label CORTEX_LEFT ...`.
  The `hcp_utils` Python package also bundles an MMP1.0 32k fs_LR parcellation for convenience.
- **Sphere surfaces** for spin tests: `S1200.{L,R}.sphere.32k_fs_LR.surf.gii` from the HCP S1200
  group-average release (ConnectomeDB, free registration) or `neuromaps.datasets.fetch_fslr()`.
- Parcel centroids + great-circle/geodesic distances:
  `python scripts/download_data.py geometry --atlas glasser` (uses `maps.parcel_centroids_from_gifti`;
  for true surface geodesics use `brainsmash.workbench.geo.cortex` with `wb_command`).

## 3. Cortical maps (open via neuromaps unless stated)

```bash
python scripts/download_data.py maps      # fetches the neuromaps annotations below and parcellates them
```

| Map | Source | neuromaps key |
|---|---|---|
| T1w/T2w myelin (HCP S1200 group average) | Glasser & Van Essen, 2011 | `('hcps1200','myelinmap','fsLR','32k')` |
| Cortical thickness (HCP S1200) | HCP | `('hcps1200','thickness','fsLR','32k')` |
| Principal FC gradient | Margulies et al., 2016 | `('margulies2016','fcgradient01','fsLR','32k')` |
| MEG band power (alpha, beta, ...) | Shafiei et al. 2022 / HCP MEG | `('hcps1200','megalpha','fsLR','4k')` etc. |
| Sensorimotor-association axis | Sydnor et al., 2021 | `('sydnor2021','SAaxis','fsLR','32k')` (if present in your neuromaps version) |
| Evolutionary / developmental expansion | Hill et al., 2010 | `('hill2010','evoexp','fsLR','164k')` |

Maps that are not in neuromaps (e.g. adolescent thickness change from
Whitaker et al., 2016; structural covariance from Romero-Garcia et al., 2018)
are re-derived from the original open data (NSPN via PIs; HCP-YA) or taken
from the authors' supplementary tables and saved as `region,value` CSVs.

## 4. HCP-Aging age-effect maps (NDA Data Use Certification)

HCP-Aging (NDA collection 2847) needs an institutional DUC. Download the
MSMAll-registered `thickness` / `MyelinMap` pscalar or dscalar files for all
participants with `downloadcmd` (nda-tools), then compute per-parcel age
slopes (linear + quadratic, adjusted for sex and site) with
`scripts/compute_age_effect_maps.py` (to be written; the parcel loader is
`maps.load_dscalar_with_dlabel`). Store only the group-level parcel maps under
`data/maps/`; participant-level HCP-A data never leave the NDA-approved
environment.

## 5. Gene sets (open)

```bash
python scripts/download_data.py genesets            # go-basic.obo + NCBI gene2go (human, taxid 9606)
```

Also curate the literature gene lists used by the classic findings (HSE
genes from Zeng et al., 2012 via Krienen et al., 2016; brain-specific genes
from Burt et al., 2018; oligodendrocyte / synaptic sets from Whitaker et al.,
2016; cell-type markers from Seidlitz et al., 2020) as one-symbol-per-line
files in `data/genesets/`; cite the source table in each file header.
