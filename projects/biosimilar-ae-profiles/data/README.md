# Data acquisition — biosimilar-ae-profiles

Nothing here is committed. All sources are public.

## Expected layout

```
data/
  raw/
    reports_<inn>_<brand>.jsonl      # --sample output (attributed, flattened reports)
    reports_<inn>_inn_only.jsonl
    labels_<brand>.json              # SPL sections per brand
  bulk/                              # openFDA drug/event partitions (json.zip)
  faers_ascii/                       # optional quarterly ASCII zips
  purple_book/purple_book_<YYYY-MM>.csv
  denominators/
    sdud_<year>.csv                  # Medicaid State Drug Utilization Data
  processed/
    family_reports.parquet           # all reports with a catalogued family, attributed
    attribution_conflicts.csv
```

## 1. openFDA (open; free key optional)

```
export OPENFDA_API_KEY=...            # https://open.fda.gov/apis/authentication/
python scripts/download_data.py --sample --families ADALIMUMAB INFLIXIMAB PEGFILGRASTIM --per-product 300
```
The sample pulls, per family, the originator brand and each launched
biosimilar brand (search on `patient.drug.openfda.brand_name` OR
`patient.drug.medicinalproduct`), the INN-only reports, and each brand's
latest label from `drug/label` (sorted by `effective_time:desc`).

Full database: stream the bulk partitions (`https://api.fda.gov/download.json`
→ `results.drug.event.partitions[*].file`) through
`biosim_ae.products.attribute_report`, keeping reports with non-empty `inns`.

Field notes: `patient.drug.openfda.brand_name` / `generic_name` /
`substance_name` / `manufacturer_name` are openFDA's harmonised fields and are
absent for unmapped entries; `medicinalproduct` is the verbatim reported name
and is always present. `reporttype` 1 = spontaneous, 2 = report from study,
3 = other, 4 = not available. `primarysource.qualification` 1 physician,
2 pharmacist, 3 other HCP, 4 lawyer, 5 consumer.

## 2. FAERS quarterly ASCII (open)

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html — needed for
`CASEID`/`CASEVERSION` deduplication, `RPSR_COD` (report source: FGN, HP,
CSM = company/manufacturer, etc.) and `DRUG.NDA_NUM` (BLA numbers for
biologics, the most reliable product identifier when present).

## 3. Purple Book (open)

https://purplebooksearch.fda.gov/ → "Download" gives the monthly CSV of all
licensed biologics with BLA number, biosimilar/interchangeable flags and
reference product. Reconcile `biosim_ae.products.FAMILIES` against it before
any analysis (launch months are *not* in the Purple Book; use company press
releases / trade press and record the source in `purple_book/launches.csv`).

## 4. Labels (open)

`scripts/download_data.py --sample` fetches the latest SPL per brand. For
label version history use DailyMed:
https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
(full archive) or the DailyMed REST API `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=<brand>`.

## 5. Denominators (open)

- Medicaid State Drug Utilization Data (per NDC, quarterly, all states):
  https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data
  Map NDC → product via openFDA `drug/ndc` or the NDC directory.
- Medicare Part B drug spending (HCPCS-level, e.g. Q5103 Inflectra, Q5104
  Renflexis, J1745 Remicade): https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug/medicare-part-b-spending-by-drug

## Sizes

- Family-filtered FAERS reports: a few hundred thousand (< 2 GB flattened).
- Sample mode: < 30 MB.
