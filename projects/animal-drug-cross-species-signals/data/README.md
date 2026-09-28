# Data acquisition

All sources are open. Only this README is committed.

## 0. API key (optional, raises limits to 240 req/min)

```bash
export OPENFDA_API_KEY=...
```

## 1. CVM animal & veterinary adverse events (openFDA)

```bash
python scripts/download_data.py --animal --start 2008-01-01          # full (~1.3M reports, hours)
python scripts/download_data.py --sample                              # ~500 dog + cat reports
python scripts/download_data.py --animal --species Dog Cat --start 2015-01-01
```

The script pages `https://api.fda.gov/animalandveterinary/event.json` in 30-day windows of
`original_receive_date` (each window stays under the 25,000-record `skip` limit), writes raw
records to `data/animal/raw/<YYYYMMDD>_<YYYYMMDD>.jsonl`, and flattens them to
`data/animal/flat/animal_events.parquet` (one row per report x drug x reaction) with columns:

```
report_id, unique_aer_id_number, receive_date, onset_date, species, breed, gender, age_value,
age_unit, weight_min, weight_max, weight_unit, serious_ae, primary_reporter, outcome,
drug_index, brand_name, active_ingredient, ingredient_norm, dose_value, dose_unit, route,
dosage_form, used_according_to_label, veddra_term, veddra_code, veddra_version, n_affected
```

Field names follow the openFDA searchable-fields page for this endpoint
(https://open.fda.gov/apis/animalandveterinary/event/searchable-fields/); the flattener is
defensive about missing keys. Notable fields: `animal.species` (Dog, Cat, Horse, Cattle, ...),
`animal.weight.{min,max,unit}`, `drug[].active_ingredients[].{name,dose}`,
`reaction[].{veddra_version,veddra_term_code,veddra_term_name,number_of_animals_affected,accuracy}`,
`outcome[].{medical_status,number_of_animals_affected}`, `serious_ae`, `primary_reporter`,
`original_receive_date`, `onset_date`, `health_assessment_prior_to_exposure`, `type_of_information`.

## 2. FAERS human counts (openFDA)

```bash
python scripts/download_data.py --faers --ingredients data/shared_ingredients.txt
```

For each ingredient: top 300 PTs (`count=patient.reaction.reactionmeddrapt.exact`) and daily
counts by `receivedate` for each (ingredient, PT), plus ingredient totals, PT totals and N.
Files under `data/faers/`. Only primary-suspect drugs (`drugcharacterization:1`).

## 3. VeDDRA term list

EMA publishes the combined VeDDRA list (LLT/PT/HLT/SOC) as a spreadsheet; search
"VeDDRA list of clinical terms" on https://www.ema.europa.eu and save as
`data/veddra/veddra_terms.csv` (columns: `soc, hlt, pt, llt, code`). The organ-system classifier in
`xspecies_pv.term_mapping` works without it; the SOC column is used to refine bucket assignment.

## 4. MedDRA (licensed)

Place `pt.asc`, `soc.asc`, `hlt.asc`, `hlt_pt.asc`, `hlgt_hlt.asc`, `soc_hlgt.asc` from your
licensed distribution under `data/meddra/<version>/`. Never commit.

## 5. Approval dates and labelled doses

- Human: `python scripts/download_data.py --drugsfda` -> `data/drugsfda/applications.jsonl`.
- Veterinary: Animal Drugs @ FDA (https://animaldrugsatfda.fda.gov) and the Green Book data
  files (https://www.fda.gov/animal-veterinary/products/approved-animal-drug-products-green-book);
  save the product table as `data/greenbook/products.csv` (columns: application number, ingredient,
  species, approval date, dosage form). Labelled mg/kg doses are entered manually for the shared
  ingredients into `data/greenbook/labelled_doses.csv` (`ingredient, species, mg_per_kg_low, mg_per_kg_high, source`).

## Expected layout

```
data/
  README.md
  shared_ingredients.txt
  animal/raw/*.jsonl
  animal/flat/animal_events.parquet
  faers/<ingredient>__<pt>.csv, <ingredient>__ALL.csv, _PT__<pt>.csv, _N.csv
  veddra/veddra_terms.csv
  meddra/<version>/*.asc
  drugsfda/applications.jsonl
  greenbook/products.csv, labelled_doses.csv
outputs/
  crosswalk_organ_system.csv
  signal_table_dog.parquet, signal_table_cat.parquet, signal_table_human.parquet
  concordance.csv, lead_lag.csv
```
