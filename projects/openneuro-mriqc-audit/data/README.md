# Data acquisition: openneuro-mriqc-audit

Everything used here is public and needs no registration. Nothing under
`data/` is committed.

Expected layout:

```
data/
  openneuro/
    datasets.csv                 # one row per dataset (GraphQL listing)
    subject_metadata.csv         # participantId/age/sex/group from summary.subjectMetadata
    participants/<dsid>.tsv      # raw participants.tsv per dataset (optional, richer columns)
  mriqc_webapi/
    T1w.jsonl  bold.jsonl        # raw records (append-only cache)
    T1w.parquet bold.parquet     # flattened + de-duplicated tables
  openneuro_git/<dsid>/          # shallow git clones (symlinks only, ~MBs) for MD5 linkage
  linked/
    iqms_linked.parquet          # IQM records joined to dataset / subject / age
```

## 1. OpenNeuro dataset and participant metadata (GraphQL)

Endpoint: `https://openneuro.org/crn/graphql` (no auth for public datasets).
The client in `src/mriqc_audit/openneuro_client.py` pages through
`datasets(first: 100, after: <cursor>)` and reads, per dataset,
`latestSnapshot.summary.subjectMetadata { participantId age sex group }`
(server-side parse of `participants.tsv`) and the curated `metadata`
(`studyDomain`, `species`, `studyLongitudinal`, `associatedPaperDOI`).

```bash
python scripts/download_data.py openneuro --sample 20      # first 20 datasets
python scripts/download_data.py openneuro                  # all (~1,000+ datasets, a few minutes)
python scripts/download_data.py participants --sample 20   # raw participants.tsv files
```

An API key (OpenNeuro account -> "Obtain an API key") is only needed for
private datasets; set `OPENNEURO_API_KEY` if you have one.

Scanner information is **not** in the GraphQL summary; it comes from the
JSON sidecars (`Manufacturer`, `ManufacturersModelName`,
`MagneticFieldStrength`) which the MRIQC record already carries in
`bids_meta`. For datasets without WebAPI records, sidecars can be fetched
individually from the `urls` returned by `snapshot_files()` (they are tiny).

## 2. MRIQC WebAPI (crowdsourced IQMs)

Root: `https://mriqc.nimh.nih.gov/api/v1/{T1w,T2w,bold}`; python-eve pagination
(`?page=k&max_results=1000`), optional `?where={"bids_meta.MagneticFieldStrength":3}`.

```bash
python scripts/download_data.py mriqc --modality T1w --sample 3    # 3 pages (3,000 records)
python scripts/download_data.py mriqc --modality T1w               # everything (hundreds of thousands of records)
python scripts/download_data.py mriqc --modality bold
```

Records are appended to `data/mriqc_webapi/<modality>.jsonl` and flattened to
parquet. `provenance.version` is kept: IQM definitions changed across MRIQC
releases, so analyses stratify by major version.

## 3. Linking IQM records to OpenNeuro datasets (MD5)

WebAPI records do not carry a dataset id and their `subject_id` is hashed, but
`provenance.md5sum` is the MD5 of the input NIfTI bytes. OpenNeuro stores files
under git-annex `MD5E-s<size>--<md5>` keys, visible in the symlink targets of a
plain git clone (no image download):

```bash
python scripts/download_data.py annex-index --sample 20     # clones + indexes 20 datasets
# equivalent manual steps:
git clone --depth 1 https://github.com/OpenNeuroDatasets/ds000001 data/openneuro_git/ds000001
ls -l data/openneuro_git/ds000001/sub-01/anat/          # -> ../../.git/annex/objects/.../MD5E-s...--<md5>.nii.gz
```

`annex_md5_index()` walks the clone, `link_iqms_to_openneuro()` joins on MD5.
Caveats: a file re-uploaded in a later snapshot gets a new MD5 (index all
snapshots by walking `git log --all` if coverage matters); an MRIQC run on a
locally re-compressed copy will not match. Expect partial coverage; the
un-linked records still feed the normative charts (scanner/field-strength
strata come from `bids_meta`).

For datasets without WebAPI coverage, run MRIQC yourself (`docker run
nipreps/mriqc:<ver> <bids_dir> <out_dir> participant --no-sub`) on a stratified
sample; `datalad clone` + `datalad get sub-*/anat/*_T1w.nii.gz` fetches only
what is needed. Do not upload IQMs from consented-restricted datasets to the
WebAPI unless the dataset licence allows it (OpenNeuro data are CC0, so this is
normally fine).

## 4. Optional external anchors

- ABIDE-II / ADHD-200 (open, INDI): extra pediatric/clinical IQM distributions.
- MRIQC's own expert-rated ABIDE subset (Esteban et al., 2017): rater labels
  to calibrate thresholds.
