# Data acquisition — mriqc-scannability-equity

All inputs are open (OpenNeuro, CC0) except the optional ABCD/HBN replication. Nothing under `data/` is committed.

## 1. Dataset index and participants tables (small, scripted)

```bash
pip install requests
python scripts/download_data.py --index                 # all datasets with anat/func → data/index/datasets.csv
python scripts/download_data.py --participants          # participants.tsv/json + dataset_description.json per dataset
python scripts/download_data.py --participants --sample # first 10 datasets only
```

The script uses the OpenNeuro GraphQL endpoint (`https://openneuro.org/crn/graphql`) to page through datasets and the snapshot file listing to fetch `participants.tsv`, `participants.json`, `dataset_description.json`, and — when present — `derivatives/mriqc/group_T1w.tsv` / `group_bold.tsv`. Raw files are fetched by their snapshot URL (`https://openneuro.org/crn/datasets/<ds>/snapshots/<tag>/files/<path>`).

Layout:

```
data/index/datasets.csv                       # id, tag, name, n_subjects, modalities, scanner fields
data/openneuro/<dsid>/participants.tsv
data/openneuro/<dsid>/participants.json
data/openneuro/<dsid>/dataset_description.json
data/openneuro/<dsid>/derivatives/mriqc/group_T1w.tsv   # if shipped
```

## 2. Raw images for MRIQC (large)

Use one of:

```bash
# openneuro-py (pip install openneuro-py)
openneuro-py download --dataset ds000030 --target-dir data/openneuro/ds000030 --include "sub-*/anat/*T1w.nii.gz" --include "sub-*/func/*bold.nii.gz"

# DataLad
datalad install https://github.com/OpenNeuroDatasets/ds000030.git data/openneuro/ds000030
datalad get data/openneuro/ds000030/sub-*/anat/*T1w.nii.gz

# AWS S3 (anonymous)
aws s3 sync --no-sign-request s3://openneuro.org/ds000030 data/openneuro/ds000030 --exclude "*" --include "sub-*/anat/*T1w.nii.gz"
```

`python scripts/download_data.py --raw ds000030 --modality anat` wraps the S3 route (requires `boto3`).

## 3. MRIQC

```bash
docker pull nipreps/mriqc:24.0.2
docker run --rm -v $PWD/data/openneuro/ds000030:/data:ro -v $PWD/data/mriqc/ds000030:/out \
    nipreps/mriqc:24.0.2 /data /out participant --no-sub -m T1w bold --nprocs 8
docker run --rm -v $PWD/data/openneuro/ds000030:/data:ro -v $PWD/data/mriqc/ds000030:/out \
    nipreps/mriqc:24.0.2 /data /out group
# classifier prediction on T1w IQMs
docker run --rm -v $PWD/data/mriqc/ds000030:/out nipreps/mriqc:24.0.2 mriqc_clf --load-classifier -X /out/group_T1w.tsv
```

Outputs used: `data/mriqc/<dsid>/group_T1w.tsv`, `group_bold.tsv`, and the classifier CSV. Record the MRIQC version per dataset (`--version` in the index).

## 4. MRIQC Web-API (normative IQM distributions; no demographics)

```bash
curl "https://mriqc.nimh.nih.gov/api/v1/T1w?max_results=1000&page=1" -o data/webapi/T1w_page1.json
```

`scripts/download_data.py --webapi T1w --pages 5` pages through the API.

## 5. Optional replication cohorts

- ABCD: NDA Data Use Certification; download `abcd_smrip*`/`mriqcrp*` tables (MRIQC-derived QC metrics are provided) and demographics via `downloadcmd`. Place under `data/abcd/`.
- Healthy Brain Network: sign the HBN data usage agreement; phenotypic data via LORIS, imaging via S3 (`s3://fcp-indi/data/Projects/HBN/`).

## 6. Sample mode without network

`python scripts/download_data.py --simulate` writes a synthetic corpus (`data/sample/corpus.csv`) from `scannability.simulate` for testing the modelling code.
