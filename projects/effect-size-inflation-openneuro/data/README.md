# Data acquisition

Nothing in this folder is committed. Everything below is re-creatable.

## 1. OpenNeuro dataset index (open, no key)

The GraphQL endpoint is `https://openneuro.org/crn/graphql`. The downloader pages through
`datasets(first: 100, after: <cursor>)` and keeps the latest snapshot summary.

```bash
cd projects/effect-size-inflation-openneuro
python scripts/download_data.py --out data/openneuro --sample 50     # first 50 datasets
python scripts/download_data.py --out data/openneuro                 # everything
python scripts/download_data.py --introspect                         # print the live schema for Summary/Description
```

Output: `data/openneuro/datasets.csv` with one row per dataset:
`dataset_id, name, created, latest_tag, snapshot_created, n_subjects, n_sessions, modalities, tasks,
total_files, size_bytes, dataset_doi, reference_dois, n_authors`.

If the endpoint is unreachable (some institutional proxies block it), use the GitHub mirrors, which
hold every dataset's `participants.tsv` and `dataset_description.json` in plain git (branch `master`):

```bash
python scripts/download_data.py --out data/openneuro --fallback-github --sample 50
```

This enumerates `ds000001 ... ds00NNNN` on `https://raw.githubusercontent.com/OpenNeuroDatasets/<id>/master/`
and counts `participants.tsv` rows. It is slower (one request per dataset) and rate-limited by GitHub
(60 unauthenticated requests/hour; set `GITHUB_TOKEN` to raise it to 5000/hour).

Manual verification of n for a few datasets: open `https://openneuro.org/datasets/ds000001` and compare
the "Participants" count with `n_subjects`.

## 2. Linked papers (open)

`reference_dois` and `dataset_doi` are parsed from `dataset_description.json`. Resolve them with
NCBI E-utilities (no key needed for < 3 requests/s; set `NCBI_API_KEY` for 10/s):

```bash
python scripts/download_data.py --resolve-dois data/openneuro/datasets.csv --out data/papers
```

This writes `data/papers/doi_to_pmid.csv` (via `esearch` on `[DOI]`) and, for PMC open-access papers,
`data/papers/pmc/<PMCID>.xml` (via `efetch`). Papers that are not OA must be read through your library;
record extracted effects in `data/papers/effects_annotated.csv` with the columns documented in
`src/es_inflation/effects.py::ANNOTATION_COLUMNS`.

## 3. NeuroVault statistic maps (open)

```bash
python scripts/download_data.py --neurovault data/openneuro/datasets.csv --out data/neurovault
```

Queries `https://neurovault.org/api/collections/?DOI=<doi>` for each DOI and lists the collection's
images (`/api/collections/<id>/images/`). Download unthresholded group maps only when needed for the
re-analysis comparison (`--download-maps`); each map is 1-5 MB.

## 4. OpenNeuro Derivatives (fMRIPrep outputs; open, large)

Only for the re-analysis subset (20-30 datasets). Use DataLad:

```bash
pip install datalad datalad-osf
datalad clone https://github.com/OpenNeuroDerivatives/ds000102-fmriprep data/derivatives/ds000102-fmriprep
cd data/derivatives/ds000102-fmriprep
datalad get sub-*/func/*desc-preproc_bold.nii.gz sub-*/func/*desc-confounds_timeseries.tsv
```

Or directly from S3 (`s3://openneuro-derivatives/fmriprep/<dsid>-fmriprep/`, public, no credentials;
`aws s3 sync --no-sign-request`). Expect 5-50 GB per dataset.

Raw BIDS data (events.tsv needed for the GLM design) come from OpenNeuro itself:

```bash
pip install openneuro-py
openneuro-py download --dataset ds000102 --include "sub-*/func/*events.tsv" --target-dir data/raw/ds000102
```

## 5. Large-n reference effects

- HCP S1200 group-average task contrasts: free registration at https://db.humanconnectome.org (accept the
  Open Access Data Use Terms), then download "HCP_S1200_GroupAvg_v1" (about 1 GB).
- UK Biobank task-fMRI group summaries require an approved application; not needed for the core analyses.

## Expected layout

```
data/
  openneuro/datasets.csv
  openneuro/raw_pages/*.json          # cached GraphQL pages
  papers/doi_to_pmid.csv
  papers/pmc/*.xml
  papers/effects_annotated.csv        # hand-annotated effects (the core table)
  neurovault/collections.csv
  neurovault/maps/*.nii.gz
  derivatives/<dsid>-fmriprep/
  raw/<dsid>/
  reference/HCP_S1200_GroupAvg_v1/
```
