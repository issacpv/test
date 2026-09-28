# Data acquisition

Nothing in this folder is committed. Expected layout after the steps below:

```
data/
  registry.csv                 # tES datasets found on OpenNeuro (built by --search, hand-verified)
  openneuro/
    ds003670/                  # BIDS datasets (one folder per accession)
    ds006126/
    ds000221/                  # MPI-LEMON (normative T1s)
  ixi/
    IXI-T1/*.nii.gz
  simnibs/
    <dataset>/<sub>/m2m_<sub>/  # charm outputs
    <dataset>/<sub>/fields/     # *_TDCS_1_scalar.msh, ROI summaries
  derived/
    dose_metrics.csv           # one row per subject x montage
    behaviour.csv              # one row per subject: effect, covariates
```

## 1. Build the tES registry (OpenNeuro GraphQL)

```
python scripts/download_data.py --search
```

This pages through the OpenNeuro GraphQL endpoint (`https://openneuro.org/crn/graphql`), keeps datasets whose name/README mention tDCS/tACS/tRNS/tES/"transcranial electrical", and writes `data/registry.csv` with accession, name, subject count, modalities and whether a T1w is present. Hand-verify each row (montage, current, outcome measure) before use; add columns `montage`, `current_mA`, `target_roi`, `outcome_column`.

## 2. Download BIDS datasets (open, no credentials)

Option A, openneuro-py (pip):

```
openneuro-py download --dataset ds003670 --target-dir data/openneuro/ds003670
openneuro-py download --dataset ds006126 --target-dir data/openneuro/ds006126
# only anat + participants for a subject:
openneuro-py download --dataset ds000221 --include sub-010002/ses-01/anat --include participants.tsv --target-dir data/openneuro/ds000221
```

Option B, DataLad (lazy, recommended for LEMON's size):

```
datalad clone https://github.com/OpenNeuroDatasets/ds000221.git data/openneuro/ds000221
cd data/openneuro/ds000221 && datalad get sub-*/ses-01/anat/*T1w.nii.gz
```

`python scripts/download_data.py --sample` fetches `participants.tsv`, `dataset_description.json` and one subject's anat folder for each registry dataset via openneuro-py (small; a few hundred MB at most).

## 3. IXI (open, CC BY-SA 3.0)

Download `IXI-T1.tar` from https://brain-development.org/ixi-dataset/ and the demographic spreadsheet `IXI.xls`; extract into `data/ixi/IXI-T1/`. The script prints the exact URLs (`--ixi`), but the tarball (~4.5 GB) is not fetched automatically.

## 4. Head models and simulations (SimNIBS 4)

```
charm sub-01 data/openneuro/<ds>/sub-01/anat/sub-01_T1w.nii.gz   # add T2 if present
python scripts/simulate_montage.py --subject sub-01 --montage M1-SO --current 1.0   # to be written; see README Methods step 2
```

Store outputs under `data/simnibs/<dataset>/<sub>/`. Fields are simulated at 1 mA and scaled in code.

## 5. Spatial-null surfaces

`neuromaps` fetches fsaverage spheres on first use (`neuromaps.datasets.fetch_fsaverage`), cached under `~/neuromaps-data`.
