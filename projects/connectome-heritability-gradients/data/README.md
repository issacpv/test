# Data acquisition — `connectome-heritability-gradients`

Nothing under `data/` is committed. Target layout:

```
data/
├── hcp/
│   ├── unrestricted.csv                 # ConnectomeDB behavioural export (open)
│   ├── restricted.csv                   # Family_ID, ZygosityGT, Mother_ID, Father_ID, Age_in_Yrs (DUA)
│   ├── twins.txt                        # subject IDs of all MZ/DZ twins + siblings (derived)
│   └── <subject>/
│       ├── rfMRI_REST*/*.dtseries.nii   # ICA-FIX, MSMAll (open, S3)
│       ├── rfMRI_REST*/*.Schaefer400.ptseries.nii   (you create with wb_command)
│       └── T1w/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}
├── sc/<subject>_schaefer400_sift2.csv   # structural connectomes (you create with MRtrix3)
├── parcellations/                       # Schaefer dlabel + MNI volume + info txt
├── surfaces/                            # fsLR-32k sphere GIFTIs for spin tests (from HCP pipelines templates)
├── ahba/expression_Schaefer400.csv      # abagen output (public)
└── abcd/                                # optional replication (NDA)
```

## 1. HCP S1200 — open imaging (free registration) and restricted family data (DUA)

1. Register at <https://db.humanconnectome.org>, accept the Open Access terms, and create AWS keys
   (profile → *Amazon S3 access*). Export `HCP_AWS_ACCESS_KEY_ID` / `HCP_AWS_SECRET_ACCESS_KEY`.
2. Download the *unrestricted* behavioural CSV (Subject, Gender, Age bins, NIH Toolbox, …).
3. Apply for **Restricted Data** — mandatory for this project:
   <https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage>.
   The restricted CSV provides `Family_ID`, `Mother_ID`, `Father_ID`, `ZygosityGT` (genotyped, preferred),
   `ZygositySR` (self-report) and `Age_in_Yrs`. `conn_h2.heritability.build_twin_pairs` turns this into
   MZ / DZ / non-twin sibling pairs. HCP S1200 contains roughly 130–150 genotype-confirmed MZ pairs and
   ~70–90 DZ pairs with complete 4-run rest data; verify counts from your own export.
4. Rest fMRI (per subject 4 runs × ~1 GB):
   ```bash
   python scripts/download_data.py --hcp-rest --subjects data/hcp/twins.txt
   wb_command -cifti-parcellate <dtseries> data/parcellations/Schaefer2018_400Parcels_7Networks_order.dlabel.nii COLUMN <ptseries>
   ```
5. Diffusion (per subject ~4.5 GB) for structural connectomes:
   ```bash
   python scripts/download_data.py --hcp-diffusion --subjects data/hcp/twins.txt
   ```
   Then MRtrix3: `dwi2response dhollander → dwi2fod msmt_csd → 5ttgen → tckgen -act -seed_dynamic 10M → tcksift2 →
   tck2connectome -scale_invnodevol` with the Schaefer-400 volume registered to T1w. Budget ~3 CPU-h/subject.
   Short-cut for a pilot: use a group-consensus SC (e.g. from the ENIGMA toolbox `load_sc`) for coupling
   estimates — but twin heritability needs *individual* SC.
6. Retest subset (45 subjects with a second full session) for reliability ceilings: same S3 layout under
   `HCP_Retest/`.

## 2. Allen Human Brain Atlas (public)

```bash
python scripts/download_data.py --parcellations
python scripts/download_data.py --abagen --atlas data/parcellations/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
```
`abagen.get_expression_data(atlas, lr_mirror='bidirectional', missing='interpolate', norm_matched=True)` downloads the
six donors' microarray data (~4 GB) to `~/abagen-data` (`ABAGEN_DATA` env var to relocate). Only two donors have
right-hemisphere samples; `lr_mirror` addresses this. Record the abagen version and all options in the paper.

## 3. Surfaces for spin tests

Spin tests need parcel centroids on the **sphere**. Use the fsLR-32k sphere GIFTIs distributed with the HCP
pipelines (`global/templates/standard_mesh_atlases/L.sphere.32k_fs_LR.surf.gii`) or `neuromaps.datasets.fetch_fslr()`
(`sphere` key). `conn_h2.spatial_nulls.spherical_coords_from_labels` computes centroids from a dlabel + sphere.

## 4. ABCD replication (optional, NDA DUC)

ABCD includes ~ 400–450 twin pairs with usable rest fMRI at baseline (four twin-hub sites); the genetic-relatedness
table (`acspsw03`) gives zygosity from genotypes. Download with `nda-tools` (`downloadcmd -dp <package>`) after
building a package from the ABCD-BIDS rsfMRI derivatives and the phenotype tables. Set `NDA_USERNAME`/`NDA_PASSWORD`.

## 5. Checks

```bash
python -c "import pandas as pd, sys; sys.path.insert(0,'src'); from conn_h2.heritability import build_twin_pairs; \
df=pd.read_csv('data/hcp/restricted.csv'); print(build_twin_pairs(df).summary())"
```
