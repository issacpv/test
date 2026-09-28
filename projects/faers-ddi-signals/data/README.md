# Data acquisition — faers-ddi-signals

Nothing in this folder is committed. Every file below is re-creatable from
public sources.

## Expected layout

```
data/
  raw/                      # --sample output (JSONL, one flattened report per line)
    pair_<a>_<b>.jsonl
    drug_<name>.jsonl
    interacting_role.jsonl
    counts_receivedate_{female,male}.jsonl
  bulk/                     # openFDA quarterly JSON partitions (drug-event-0001-of-00XX.json.zip)
  faers_ascii/              # optional FAERS quarterly ASCII zips (faers_ascii_2024q1.zip ...)
  reference/
    onc_high_priority_ddi.csv       # columns: drug_a, drug_b, event_class
    crediblemeds_qtdrugs.csv        # columns: drug, category (Known/Possible/Conditional)
    fda_cyp_tables.csv              # columns: enzyme, role (inhibitor/inducer/substrate), strength, drug
    twosides_subset.csv             # optional legacy comparator
  processed/
    reports.parquet           # flattened, deduplicated report-level table
    tables_all.parquet        # n_ijk per (pair, event)
    tables_female.parquet / tables_male.parquet
```

## 1. openFDA (open, no registration; free key recommended)

1. Get a key at https://open.fda.gov/apis/authentication/ and export it:
   `export OPENFDA_API_KEY=...` (240 requests/min instead of 40).
2. Sample pull (about 30 requests, a few minutes):
   `python scripts/download_data.py --sample --per-query 500 --start 2015-01-01 --end 2026-06-30`
3. Full database: use the bulk manifest instead of paging the API.
   ```
   curl -s https://api.fda.gov/download.json > data/download_manifest.json
   python - <<'EOF'
   import json, subprocess, pathlib
   m = json.load(open("data/download_manifest.json"))["results"]["drug"]["event"]
   pathlib.Path("data/bulk").mkdir(parents=True, exist_ok=True)
   for p in m["partitions"]:
       subprocess.run(["wget", "-c", "-P", "data/bulk", p["file"]], check=True)
   EOF
   ```
   Each partition is a zipped JSON `{"results": [...]}` with the same schema as
   the API; stream it through `faers_ddi.openfda.flatten_record`.

Key field semantics: `patient.drug.drugcharacterization` 1 = suspect,
2 = concomitant, 3 = interacting; `patient.patientsex` 1 = male, 2 = female;
`patient.patientonsetageunit` 800 decade ... 805 hour (see `openfda.py`).

## 2. FAERS quarterly ASCII files (open)

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html (2012Q4 onward;
legacy AERS 2004-2012Q3 on the same page). Needed for (a) proper case
deduplication via `CASEID`/`CASEVERSION` (keep the highest version, drop
`DELETED` cases) and (b) the `DRUG.ROLE_COD` field (`PS`, `SS`, `C`, `I`) that
openFDA collapses into `drugcharacterization`. Unzip each quarter into
`data/faers_ascii/<year>q<n>/`.

## 3. Reference sets (all free)

| Set | Use | Access | Where |
|---|---|---|---|
| ONC high-priority DDI list (Phansalkar et al., 2012, *JAMIA*) | clinical positives (~15 classes of pairs) | open (paper supplement) | https://doi.org/10.1136/amiajnl-2012-000935 |
| CredibleMeds QTdrugs | QT-prolonging drugs (Known / Possible / Conditional risk); pairs of two "Known" drugs = positives for QT-class events | free registration | https://www.crediblemeds.org/ |
| FDA Drug Development and Drug Interactions tables | CYP/transporter substrates, inhibitors, inducers with strength | open | https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers |
| DrugBank DDI | broad positive list | academic licence (free for non-commercial, application) | https://go.drugbank.com/ |
| TWOSIDES (Tatonetti et al., 2012, *Sci Transl Med*) | legacy FAERS-derived DDI signals for comparison | open | https://nsides.io/ |
| MedDRA | PT -> HLT/SOC grouping for event classes | licence (free for academic use via MSSO subscription) | https://www.meddra.org/ |

Save each as CSV in `data/reference/` with the column names listed in the
layout above; `faers_ddi.reference_sets.load_pair_csv` reads them.

## 4. Drug-name harmonisation

`openfda.generic_name` is missing for roughly one in six drug entries. For the
full analysis, map `medicinalproduct` strings with the DiAna dictionary
(Fusaroli et al., 2024, *Drug Saf*; released on GitHub/OSF by the authors) or
RxNorm (https://rxnav.nlm.nih.gov/, open API). Store the mapping as
`data/reference/drug_name_map.csv` (`raw_name, generic_name`).

## Sizes

- Full openFDA drug/event bulk: roughly 20 million reports, ~10 GB zipped, ~90 GB as JSON.
- Flattened report table (one row per report, list columns) as parquet: ~5-8 GB.
- `--sample` output: < 20 MB.
