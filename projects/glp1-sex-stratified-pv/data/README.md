# Data acquisition — glp1-sex-stratified-pv

Nothing here is committed. All sources are public.

## Expected layout

```
data/
  raw/
    reports_<agent>.jsonl              # --sample: flattened GLP-1 RA reports
    reports_cmp_<class>.jsonl          # active comparator classes
    counts_<agent>_{female,male}.jsonl # top-1000 PT counts per sex
    counts_background_{female,male}.jsonl
    total_background_{female,male}.jsonl
    labels_<brand>.json
  bulk/                                # openFDA drug/event partitions
  faers_ascii/
  meps/
    h233a.csv  (Prescribed Medicines, e.g. 2021)   h233.csv (Full-Year Consolidated)
  nhanes/
    RXQ_RX_L.xpt  DEMO_L.xpt  (2021-2023)  RXQ_RX_J.xpt DEMO_J.xpt (2017-2018) ...
  processed/
    cohort.parquet  background.parquet  users_by_sex_age.csv
```

## 1. openFDA (open; free key optional)

```
export OPENFDA_API_KEY=...        # https://open.fda.gov/apis/authentication/
python scripts/download_data.py --sample --per-query 400 --start 2018-01-01 --end 2026-06-30
```
Search clauses combine `patient.drug.openfda.generic_name`,
`patient.drug.medicinalproduct` (verbatim; catches compounded products with no
openFDA mapping) and `patient.drug.openfda.brand_name`. Sex-specific count
tables use `count=patient.reaction.reactionmeddrapt.exact` with
`patient.patientsex:1|2` in the search.

Full data: bulk partitions from `https://api.fda.gov/download.json`
(`results.drug.event.partitions`), streamed through
`glp1_sexpv.cohort.flatten_report`; keep exposed reports, comparator reports
and a random 10 % of the remainder as background.

## 2. FAERS quarterly ASCII (open)

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html — `DEMO`
(CASEID, CASEVERSION, SEX, AGE, AGE_COD), `DRUG` (DRUGNAME, PROD_AI, ROLE_COD,
ROUTE, DOSE_VBM — compounded products often appear as "SEMAGLUTIDE" with
pharmacy names in DRUGNAME), `INDI` (INDI_PT), `REAC`, `OUTC`, `RPSR`.

## 3. MEPS (open, no registration)

https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp
- Prescribed Medicines file per year (e.g. HC-233A for 2021, HC-239A for 2022):
  variables `DUPERSID`, `RXDRGNAM`, `RXNDC`, `RXBEGYRX`, weights `PERWTyyF`.
- Full-Year Consolidated file (HC-233 for 2021): `DUPERSID`, `SEX` (1 male,
  2 female), `AGELAST`, `PERWTyyF`, variance strata `VARSTR`, `VARPSU`.
Download the CSV or SAS transport (`.ssp`/`.xpt`; `pandas.read_sas` reads
xpt). `glp1_sexpv.denominators.flag_glp1_fills` + `users_from_fills` produce
weighted users by sex; use `VARSTR`/`VARPSU` with a survey package
(e.g. `samplics` or R `survey`) for design-based variances.

## 4. NHANES (open)

https://wwwn.cdc.gov/nchs/nhanes/ → cycle → Questionnaire → `RXQ_RX`
(`SEQN`, `RXDDRUG` generic name, `RXDUSE`) and Demographics → `DEMO`
(`RIAGENDR` 1 male 2 female, `RIDAGEYR`, weights `WTINT2YR` / `WTINTPRP`).
GLP-1 RA users are identified by `RXDDRUG` matching the name regex.

## 5. Labels (open)

`scripts/download_data.py --sample` saves the latest SPL sections per brand.
History: DailyMed `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=OZEMPIC`.

## 6. Optional second system

EudraVigilance public reports (https://www.adrreports.eu/) allow sex-stratified
line listings per substance for replication; VigiBase requires an application
to UMC.

## Sizes

- Exposed + comparator + 10 % background: ~3 M reports, ~3 GB flattened.
- MEPS/NHANES files: < 1 GB in total. Sample mode: < 50 MB.
