# Data acquisition — faers-reporting-bias

Nothing under `data/` except this file and `dsc_events_seed.csv` is committed.
All sources below are open (no registration or DUA).

## Expected layout

```
data/
├── README.md
├── dsc_events_seed.csv              # curated FDA Drug Safety Communication dates (verify each row)
├── raw/
│   ├── openfda_sample/              # --sample output (count tables, monthly series, 2x2)
│   │   ├── faers_monthly_total.csv
│   │   ├── faers_monthly_total_{female,male}.csv
│   │   ├── sex_counts.csv  reporter_counts.csv  age_counts.csv
│   │   ├── monthly/<DRUG>[_female|_male].csv
│   │   ├── reactions/<DRUG>[_female|_male].csv
│   │   └── two_by_two_sample.csv
│   ├── faers_bulk/<YYYYqN>/drug-event-XXXX-of-YYYY.json.zip   # --faers-bulk
│   ├── partd/partd_spending_by_drug.csv                        # --partd or manual CSV
│   └── meps/h248a*.csv, h248*.csv ...                          # MEPS PMED + consolidated
└── processed/                        # parquet outputs of the pipeline (ignored)
```

## 1. FAERS via openFDA (`drug/event`) — open

* Docs: https://open.fda.gov/apis/drug/event/ ; field reference:
  https://open.fda.gov/apis/drug/event/searchable-fields/
* Get a free API key (raises limits from 40 req/min & 1 000/day to 240 req/min & 120 000/day):
  https://open.fda.gov/apis/authentication/  → `export OPENFDA_API_KEY=...`
* Quick sample (count queries only, ~150 requests, a few minutes):

```bash
export OPENFDA_API_KEY=...          # optional
python scripts/download_data.py --sample --start 2010-01-01 --end 2025-06-30
```

* Full data: the API caps `skip` at 25 000, so report-level analyses use the
  quarterly bulk JSON partitions listed at https://api.fda.gov/download.json
  (~35 zipped files per quarter since 2004; ~1–2 GB per recent year).

```bash
python scripts/download_data.py --faers-bulk --years 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024
```

  Each partition is a JSON object with a `results` list of ICSRs. Use
  `faers_bias.openfda_client.flatten_faers_record` to flatten them, then
  **de-duplicate by `safetyreportid` keeping the latest `receiptdate`/version**
  (openFDA keeps follow-up versions of the same case).
* Alternatively FDA's own quarterly ASCII/XML extracts:
  https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html (needs the
  `DEMO`/`DRUG`/`REAC`/`RPSR` table joins; the `RPSR` table gives report source).

Key fields used: `receivedate`, `primarysource.qualification` (1 physician,
2 pharmacist, 3 other HCP, 4 lawyer, 5 consumer), `patient.patientsex`
(1 male, 2 female), `patient.patientonsetage(+unit)`,
`patient.drug[].openfda.generic_name`, `patient.drug[].drugcharacterization`
(1 suspect, 2 concomitant, 3 interacting), `patient.reaction[].reactionmeddrapt`,
`serious`, `occurcountry`.

## 2. Medicare Part D "Spending by Drug" (CMS) — open

* Landing page: https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug/medicare-part-d-spending-by-drug
* Data dictionary: https://data.cms.gov/resources/medicare-part-d-spending-by-drug-data-dictionary
* Columns: `Brnd_Name, Gnrc_Name, Tot_Mftr, Mftr_Name, Tot_Spndng_<YYYY>,
  Tot_Dsg_Unts_<YYYY>, Tot_Clms_<YYYY>, Tot_Benes_<YYYY>, ...` for the latest
  five calendar years (one row per brand/generic/manufacturer). Cells with < 11
  beneficiaries are suppressed.
* Download either the CSV from the page (put it at
  `data/raw/partd/partd_spending_by_drug.csv`) or via the data API: click
  "API" on the dataset page, copy the UUID from
  `https://data.cms.gov/data-api/v1/dataset/<UUID>/data`, then

```bash
python scripts/download_data.py --partd --cms-dataset-id <UUID>
```

* Companion file with age ≥ 65 breakdowns per prescriber-drug:
  "Medicare Part D Prescribers – by Provider and Drug"
  (`Tot_Clms, Tot_Benes, GE65_Tot_Clms, GE65_Tot_Benes`) — same site; useful
  for age-stratified denominators. Neither file has patient sex per drug, which
  is why MEPS is needed for sex-stratified denominators.

## 3. MEPS Household Component (AHRQ) — open

* Prescribed Medicines file (event level; one row per fill): e.g. 2023 = **HC-248A**,
  2022 = HC-239A, 2021 = HC-229A. Search "Prescribed Medicines" at
  https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp
* Full-Year Consolidated file (person level: `SEX`, `AGE23X`, `PERWT23F`,
  `VARSTR`, `VARPSU` for design-based variance): its HC number differs from the
  PMED file number — find "Full Year Consolidated Data File" for the same year
  in the listing.
* Variables used: `DUPERSID`, `RXDRGNAM` (Multum generic name), `RXNDC`,
  `RXQUANTY`, `RXDAYSUP`, `PERWT<yy>F`, `SEX`, `AGE<yy>X`.
* Denominator definition: weighted number of persons with ≥ 1 fill of
  `RXDRGNAM` in the year, by sex and age band (`faers_bias.denominators.aggregate_meps_pmed`).
  Multi-year pooling (2018-2023) is recommended for drugs with < 100 unweighted users.

```bash
python scripts/download_data.py --meps    # prints instructions; --meps-files URL1,URL2 to fetch
```

## 4. FDA Drug Safety Communications & media events — open

* Index: https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-communications
  (one page per year; each DSC has a date, product(s) and the safety issue).
* `data/dsc_events_seed.csv` holds a starter list. **Verify every row against
  the FDA page before analysis** and extend it with: FDA press announcements,
  boxed-warning additions from the Safety-related Labeling Changes database
  (https://www.accessdata.fda.gov/scripts/cder/safetylabelingchanges/), and
  media events (e.g. Google Trends peaks for the drug name, exported as CSV).
* Optional media-attention covariate: Google Trends monthly interest for the
  drug name (`pytrends`, unofficial) or GDELT article counts.

## Ethics / use

All sources are de-identified public data. FAERS reports may contain free
text; do not attempt re-identification and do not send raw narratives to
third-party LLM APIs without checking your institution's policy.
