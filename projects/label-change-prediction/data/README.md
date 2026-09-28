# Data acquisition

All sources are open. Only this README is committed.

## 0. API key (optional)

```bash
export OPENFDA_API_KEY=...   # https://open.fda.gov/apis/authentication/
```

## 1. SrLC export (label-change events, 2016-)

The SrLC database at
https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-related-labeling-changes-srlc-database-overview-updates-safety-information-fda-approved
(search UI at https://www.accessdata.fda.gov/scripts/cder/safetylabelingchanges/) offers a
download of search results. Search with no filters (or per section: "BOXED WARNING",
"WARNINGS AND PRECAUTIONS") and export to `data/srlc/srlc_export.csv` (or .xlsx -> save as CSV).
`label_change.dailymed_spl.parse_srlc_export` reads it with tolerant column matching.

## 2. DailyMed SPL history and versioned labels

```bash
python scripts/download_data.py --dailymed --drugs data/drug_list.txt   # full
python scripts/download_data.py --sample                                 # 3 drugs, history only
```

Uses DailyMed web services v2 (https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm):

- `GET /services/v2/spls.json?drug_name=<name>&pagesize=100&page=<n>` -> set IDs
- `GET /services/v2/spls/<SETID>/history.json` -> version numbers and effective dates
- `GET /services/v2/spls/<SETID>.xml` -> current SPL; older versions via the versioned download
  URL documented on the same page (the exact pattern is stored in `DailyMedClient.VERSION_URL`;
  update it if NLM changes the path).

Output: `data/dailymed/<setid>/history.json`, `data/dailymed/<setid>/v<version>.xml`.
Sections are parsed by LOINC code: 34066-1 boxed warning, 43685-7 warnings and precautions,
34070-3 contraindications, 34084-4 adverse reactions.

DailyMed also offers bulk "full release" SPL archives (https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm);
these contain current versions only and are useful for the baseline "already labelled" status.

## 3. FAERS quarterly pair counts (openFDA)

```bash
python scripts/download_data.py --faers --drugs data/drug_list.txt --pts data/pt_list.txt
```

For each drug: top reported PTs (`count=patient.reaction.reactionmeddrapt.exact`), then daily
counts by `receivedate` for each (drug, PT), for the drug total, for each PT total and for all
reports. Files: `data/faers/<drug>__<pt>.csv`, `data/faers/<drug>__ALL.csv`, `data/faers/_PT__<pt>.csv`,
`data/faers/_N.csv`. `label_change.faers_trajectories.cumulative_2x2` turns these into quarterly
2x2 tables.

## 4. Approval dates

```bash
python scripts/download_data.py --drugsfda
```

`drug/drugsfda.json` -> `data/drugsfda/applications.jsonl` (application number, products,
submissions with `submission_status_date`; the earliest approval submission gives drug age).

## 5. MedDRA

Obtain the MedDRA distribution under your organisation's licence (free for academic/non-profit
users via https://www.meddra.org/subscription) and place `pt.asc`, `hlt.asc`, `soc.asc`,
`hlt_pt.asc` etc. under `data/meddra/<version>/`. Never commit or redistribute.

## Expected layout

```
data/
  README.md
  drug_list.txt
  pt_list.txt
  srlc/srlc_export.csv
  dailymed/<setid>/history.json, v<version>.xml
  faers/<drug>__<pt>.csv ...
  drugsfda/applications.jsonl
  meddra/<version>/*.asc
outputs/
  label_events.parquet        # drug, pt, section, event_quarter, source
  pair_quarter.parquet        # drug, pt, quarter, a, b, c, d
  landmark_dataset.parquet
```
