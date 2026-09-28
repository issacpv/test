# Data acquisition — pediatric-offlabel-signals

Nothing here is committed. All sources are public.

## Expected layout

```
data/
  raw/
    peds_<drug>.jsonl              # --sample: flattened paediatric reports per panel drug
    peds_background.jsonl          # paediatric reports, all drugs
    counts_agegroup_<drug>.jsonl   # patientagegroup buckets
    labels_<drug>.json             # SPL sections + extracted floor
  bulk/
    drug-event-*.json.zip          # openFDA FAERS partitions
    drug-label-*.json.zip          # openFDA label partitions
  faers_ascii/
  reference/
    fda_pediatric_labeling_changes.csv   # from the FDA web table (see below)
    grip_reference_set.csv               # Osokogu et al. 2015 supplement
    age_floor_validation.csv             # 300 hand-labelled labels: generic, floor_years, status, annotator
  processed/
    peds_reports.parquet
    age_floors.parquet                   # generic -> floor, status, evidence, set_id, effective_time
    classified.parquet
```

## 1. openFDA (open; free key optional)

```
export OPENFDA_API_KEY=...            # https://open.fda.gov/apis/authentication/
python scripts/download_data.py --sample --per-drug 300
```
Paediatric search clause used:
`(patient.patientonsetage:[0 TO 17] AND patient.patientonsetageunit:801) OR patient.patientonsetageunit:(802 OR 803 OR 804 OR 805) OR patient.patientagegroup:(1 OR 2 OR 3 OR 4)`.
Unit codes: 800 decade, 801 year, 802 month, 803 week, 804 day, 805 hour.
Age groups: 1 neonate, 2 infant, 3 child, 4 adolescent, 5 adult, 6 elderly.

Labels: `drug/label` search `openfda.generic_name:"<name>" AND _exists_:pediatric_use`,
sorted `effective_time:desc`. Sections used: `pediatric_use`,
`indications_and_usage`, `dosage_and_administration`, `adverse_reactions`,
`boxed_warning`.

Full data: `https://api.fda.gov/download.json` lists partitions for both
`drug/event` and `drug/label`; stream the event partitions through
`peds_offlabel.age.flatten_report` (keep `pediatric == True`) and the label
partitions through `peds_offlabel.label_ages.floor_from_label`.

## 2. DailyMed (open)

Label history for dating floors at report time:
- Bulk SPL archive: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
- REST: `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/<SETID>/history.json`
  lists every version with its date; `.../spls/<SETID>/versions/<n>.xml` returns the XML,
  whose section with LOINC code 34081-0 is "Pediatric use".

## 3. FDA Pediatric Labeling Changes (open)

https://www.fda.gov/science-research/pediatrics/pediatric-labeling-changes
The table (drug, trade name, labelling change date, age range studied,
BPCA/PREA) can be copied to CSV; save as `reference/fda_pediatric_labeling_changes.csv`
with columns `generic_name, brand_name, change_date, age_range, statute`.

## 4. FAERS ASCII (open)

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html — `DEMO.AGE`,
`AGE_COD` (DEC/YR/MON/WK/DY/HR), `AGE_GRP` (N/I/C/T/A/E), `CASEID`, `CASEVERSION`.

## 5. Reference sets (open)

- GRiP paediatric reference set: Osokogu et al., 2015, *Drug Safety* 38:207-217, supplement.
- MedDRA HLGT "Medication errors" for the error-enrichment analysis (MedDRA licence, free for academic use).

## Sizes

- Paediatric FAERS: ~1.5 M reports, ~1.5 GB flattened; labels bulk ~1 GB zipped.
- Sample mode: < 40 MB.
