# Data acquisition — caers-supplement-signals

Nothing here is committed. All sources are public.

## Expected layout

```
data/
  raw/
    caers_supplements.jsonl        # --sample: industry code 54, flattened
    caers_cosmetics.jsonl          # industry code 53 (comparator stratum)
    caers_background.jsonl         # all CAERS
    counts_industry_code.jsonl  counts_supplement_reactions.jsonl  counts_supplement_outcomes.jsonl
    counts_supplement_brands.jsonl  counts_supplement_by_day.jsonl
    faers_<ingredient>.jsonl       # FAERS reports with verbatim supplement names
  bulk/
    food-event-*.json.zip          # openFDA CAERS partitions
    drug-event-*.json.zip          # openFDA FAERS partitions (for the cross-system study)
  caers_quarterly/                 # FDA CSV/XLSX files (optional cross-check)
  dsld/
    dsld_cache.json                # product name -> ingredient list (API results)
    dsld_full_export/              # optional full database export
  reference/
    livertox_hds.csv               # ingredient, likelihood_score, source_url
    fda_tainted_products.csv       # product, hidden_ingredient, category, date_listed
    negative_controls.csv          # ingredient, event
    nhanes_dsq_prevalence.csv      # ingredient, sex, age_band, weighted_users
  processed/
    caers.parquet  caers_supplements_linked.parquet  faers_supplements.parquet
```

## 1. openFDA CAERS — `food/event` (open; free key optional)

```
export OPENFDA_API_KEY=...        # https://open.fda.gov/apis/authentication/
python scripts/download_data.py --sample --per-query 1000
```
Fields: `report_number`, `date_created`, `date_started`, `outcomes[]`,
`reactions[]`, `consumer.{age,age_unit,gender}`,
`products[].{role,name_brand,industry_code,industry_name}`. Supplements are
`products.industry_code:54`. Date range searches use `date_created:[YYYYMMDD TO YYYYMMDD]`.
Count queries: `count=products.industry_code.exact`, `count=reactions.exact`,
`count=outcomes.exact`, `count=products.name_brand.exact`.

Full data (small): `https://api.fda.gov/download.json` → `results.food.event.partitions[*].file`;
stream through `caers_signals.normalize.flatten_caers`.

## 2. FDA CAERS quarterly files (open)

https://www.fda.gov/food/compliance-enforcement-food/cfsan-adverse-event-reporting-system-caers
— CSV/XLSX with the same content plus "CAERS Created Date"; useful for
deduplication checks and for any field openFDA drops.

## 3. FAERS supplement reports — `drug/event` (open)

Search verbatim names: `patient.drug.medicinalproduct:"TURMERIC"` etc. For the
full study, stream FAERS bulk partitions through
`caers_signals.normalize.flatten_faers_for_supplements` and keep `supplement == True`.

## 4. NIH DSLD (open)

- API guide: https://dsld.od.nih.gov/api-guide (search endpoint used in
  `caers_signals.normalize.dsld_search`: `<base>/search-filter?q=<name>`; label
  detail: `<base>/label/<id>`). Verify the current base URL and response schema
  on the guide page; the parser is defensive but the API has changed between versions.
- Full export (CSV/JSON of all labels and ingredients) is downloadable from the
  DSLD site for offline linkage.

## 5. Reference sets (open)

- LiverTox, "Herbal and Dietary Supplements" section: https://www.ncbi.nlm.nih.gov/books/NBK547852/
  Record each monograph's ingredient and likelihood score in `reference/livertox_hds.csv`.
- FDA Tainted Products Marketed as Dietary Supplements:
  https://www.fda.gov/drugs/medication-health-fraud/tainted-products-marketed-dietary-supplements-cder
  (downloadable table; columns product, hidden ingredient, date).
- Negative controls: nutrients/probiotics × hepatic PTs, hepatotoxins × unrelated PTs (curate in `reference/negative_controls.csv`).

## 6. NHANES Dietary Supplement Use (open)

https://wwwn.cdc.gov/nchs/nhanes/ → Dietary → `DSQTOT` (totals) and `DSQIDS`
(individual supplements with product names / ingredient codes), weights in `DEMO`.

## Sizes

- CAERS bulk: < 500 MB zipped; FAERS supplement subset: ~1 GB flattened.
- Sample mode: < 20 MB.
