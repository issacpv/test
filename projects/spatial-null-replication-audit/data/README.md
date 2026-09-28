# Data acquisition

Nothing here is committed. All inputs are public group-level maps and atlases.

## 1. Parcellations and spherical coordinates (open)

Parcel centroids are needed for spin tests and distance matrices for spectral / variogram nulls.

```bash
cd projects/spatial-null-replication-audit
python scripts/download_data.py --parcellations --out data/atlases
```

This fetches, from the CBIG GitHub repository (raw files, no key):

- `Schaefer2018_{100,200,400,1000}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv` (MNI centroids)
- the fsaverage `.annot` files for the same parcellations.

If `neuromaps` is installed the script also downloads the fsaverage (10k/41k) and fsLR (32k) sphere
surfaces (`neuromaps.datasets.fetch_atlas`) so that true spherical coordinates can be computed per
parcel (`nulls.parcel_centroids_on_sphere`). Without neuromaps, `nulls.project_to_sphere` projects MNI
centroids onto a unit hemisphere; this is an approximation and is flagged in the audit output.

## 2. Brain maps (open)

Most claims use maps distributed by neuromaps:

```bash
pip install neuromaps
python scripts/download_data.py --neuromaps --out data/maps           # lists + fetches all annotations (~1 GB)
python scripts/download_data.py --neuromaps --sample --out data/maps  # 5 annotations only
```

Maps referenced by NeuroVault image id in `claims.csv`:

```bash
python scripts/download_data.py --neurovault data/claims/claims.csv --out data/maps/neurovault
```

Gene-expression maps (for transcriptomic claims) via abagen (downloads AHBA microarray data, ~4 GB):

```bash
pip install abagen
python -c "import abagen; abagen.fetch_microarray(donors='all', data_dir='data/ahba')"
```

HCP S1200 group-average maps: register at https://db.humanconnectome.org, accept the Open Access
terms, download `HCP_S1200_GroupAvg_v1.zip` and unpack into `data/hcp/`.

ENIGMA summary maps: `pip install enigmatoolbox`, then `from enigmatoolbox.datasets import load_summary_stats`.

## 3. Claims corpus

```bash
python scripts/download_data.py --pubmed-candidates --out data/claims        # PMIDs + titles from E-utilities
```

Screen `data/claims/pubmed_candidates.csv` by hand into `data/claims/claims.csv` using the schema in
`src/spatial_null_audit/claims.py::CLAIM_COLUMNS` (a header-only template is written by the script).

## Expected layout

```
data/
  atlases/Schaefer2018_*_Centroid_RAS.csv
  atlases/*.annot
  atlases/spheres/                 # from neuromaps, optional
  maps/neuromaps/                  # neuromaps cache
  maps/neurovault/<image_id>.nii.gz
  ahba/
  hcp/HCP_S1200_GroupAvg_v1/
  claims/pubmed_candidates.csv
  claims/claims.csv
outputs/
  audit/<claim_id>.json
  calibration/*.csv
```
