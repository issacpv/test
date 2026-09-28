# Data acquisition

All sources are open. Nothing in this folder except this README is committed.

## 0. Optional API key

openFDA allows 240 requests/minute and 120,000/day with a key (40/min, 1,000/day without).
Get one at https://open.fda.gov/apis/authentication/ and export it:

```bash
export OPENFDA_API_KEY=...
```

## 1. Shortage episodes (exposure)

### 1a. openFDA Drug Shortages endpoint (primary)

```bash
python scripts/download_data.py --shortages            # all records -> data/shortages/openfda_shortages.jsonl
python scripts/download_data.py --shortages --sample   # first 100 records only
```

Records carry (field names as served by openFDA; verify against
https://open.fda.gov/apis/drug/drugshortages/ if the schema changes): `generic_name`,
`proprietary_name`, `company_name`, `presentation`, `status` (Current / Resolved / To Be
Discontinued), `initial_posting_date`, `update_date`, `update_type`, `shortage_reason`,
`therapeutic_category`, `availability`, `related_info`, `resolved_note`, `package_ndc`, `openfda.*`.

The FDA web database (https://www.accessdata.fda.gov/scripts/drugshortages/) also offers a
download of current and resolved shortages; save it as `data/shortages/fda_web_export.csv` if you
need entries older than the API history.

### 1b. ASHP / University of Utah shortage list (secondary, broader)

ASHP publishes current and resolved shortages at https://www.ashp.org/drug-shortages with
"first reported" and "resolved" dates on each bulletin page. Bulk extraction is subject to ASHP
terms of use; for research use request an extract from ASHP / UUDIS and place it at
`data/shortages/ashp_episodes.csv` with columns `generic_name,start_date,end_date,reason,dosage_form`.

## 2. FAERS outcome counts (openFDA `drug/event`)

The pipeline never downloads all FAERS reports. For each ingredient and outcome group it issues
`count=receivedate` queries:

```bash
python scripts/download_data.py --counts --drugs data/drug_list.txt          # full
python scripts/download_data.py --counts --sample                              # 5 demo drugs
```

Output: `data/faers_counts/<drug>__<group>.csv` with columns `date,count` (daily) which
`shortage_ae.shortage_panel.daily_counts_to_monthly` aggregates. Outcome groups are defined in
`shortage_ae.shortage_panel` (`MEDICATION_ERROR_PTS`, `DOSING_ERROR_PTS`, `SUPPLY_ISSUE_PTS`,
`NEGATIVE_CONTROL_PTS`, plus `all` and `serious`).

Search syntax used (see https://open.fda.gov/apis/drug/event/how-to-use-the-endpoint/):

```
patient.drug.openfda.generic_name:"HEPARIN SODIUM"+AND+patient.drug.drugcharacterization:1
  +AND+patient.reaction.reactionmeddrapt:("Wrong drug administered"+"Incorrect dose administered"+...)
```

## 3. Substitute definition (openFDA `drug/ndc`)

```bash
python scripts/download_data.py --ndc-classes
```

pulls `generic_name`, `route`, `dosage_form`, `pharm_class` for all NDC products into
`data/ndc/ndc_products.jsonl`; `shortage_panel.find_substitutes` uses the EPC class and route.

## 4. Optional: FAERS quarterly ASCII files (exact rebuild)

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html -> download each quarter's
ASCII zip into `data/faers_ascii/`. Tables DEMO/DRUG/REAC/OUTC are joined on `primaryid`,
de-duplicated on `caseid` (keep latest `fda_dt`), and PT lists applied. A helper for this is not
included in the starter code.

## 5. Optional: Medicaid State Drug Utilization Data

https://data.medicaid.gov (dataset "State Drug Utilization Data <year>") -> CSV per year into
`data/sdud/`. Used as a utilisation offset in a sensitivity analysis only.

## Expected layout

```
data/
  README.md
  drug_list.txt                  # one ingredient per line (shortage + substitute + control drugs)
  shortages/openfda_shortages.jsonl
  shortages/ashp_episodes.csv    # optional
  faers_counts/<drug>__<group>.csv
  ndc/ndc_products.jsonl
  faers_ascii/                   # optional
  sdud/                          # optional
outputs/
  panel_monthly.parquet
  event_study_*.csv
```
