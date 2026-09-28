# Data acquisition — unlabeled-adverse-event-mining

All sources are open except MedDRA (licensed; free for academic/non-profit use
via MSSO subscription). Nothing under `data/` except this README is committed.

## Layout

```
data/
├── README.md
├── raw/
│   ├── openfda_sample/               # --sample
│   │   ├── labels.csv, label_sections.csv
│   │   ├── faers_pair_counts.csv     # drug, pt, quarter, n
│   │   ├── faers_drug_totals.csv     # drug, quarter, n
│   │   ├── faers_pt_totals.csv       # pt, quarter, n (top-1000 PTs per quarter)
│   │   └── faers_all_totals.csv      # quarter, n
│   ├── labels_bulk/*.json.zip        # --labels-all (all current SPLs)
│   ├── faers_bulk/<YYYYqN>/*.json.zip
│   ├── dailymed_history.csv          # --dailymed-history
│   ├── set_ids.txt                   # your list of SPL set ids
│   ├── fda_signals_html/*.html       # quarterly FDA pages
│   ├── fda_potential_signals.csv     # parsed (quarter, product, signal_text, additional)
│   ├── srlc.csv                      # SrLC export (manual)
│   └── meddra/                       # licensed MedDRA ASCII files (never commit)
└── processed/
```

## 1. Drug labels — openFDA `drug/label` (open)

* Docs: https://open.fda.gov/apis/drug/label/ ; fields: https://open.fda.gov/apis/drug/label/searchable-fields/
* Serves the **current** version of each SPL (`set_id`, `id`, `version`,
  `effective_time`) with plain-text sections: `boxed_warning`,
  `warnings_and_cautions`, `warnings`, `precautions`, `adverse_reactions`,
  `adverse_reactions_table`, and `openfda.{generic_name, brand_name, rxcui, unii,
  application_number, product_type}`.
* Restrict to `openfda.product_type:"HUMAN PRESCRIPTION DRUG"`; many generics share
  the same text as the reference listed drug (RLD) — pool per ingredient and keep
  the RLD (application number starting with NDA/BLA) as the primary label.
* Bulk: `python scripts/download_data.py --labels-all` (~10 zipped partitions).

## 2. Label version history — DailyMed (open)

openFDA does not keep old versions. DailyMed web services (v2) provide a version
history per set id and the SPL XML of each version:

* Docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
* History: `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/<SETID>/history.json`
  (`LabelClient.fetch_dailymed_history`).
* Version XML: `.../services/v2/spls/<SETID>.xml` (add a `version` parameter for an
  archived version if supported — verify against the docs; otherwise use the
  DailyMed "SPL archive" bulk zips at https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
  which contain every historical SPL version).
* Parse with `label_client.parse_spl_xml_sections` (LOINC 34084-4 Adverse
  Reactions, 43685-7 Warnings and Precautions, 34066-1 Boxed Warning, 34071-1
  Warnings, 42232-9 Precautions).
* Be polite: ~1 request/s; ~45 000 prescription set ids × history calls is a
  multi-day job. Prioritise RLD set ids.

## 3. FDA Safety-related Labeling Changes (SrLC) database (open)

https://www.accessdata.fda.gov/scripts/cder/safetylabelingchanges/ — approved
safety labelling changes (2016→) with approval date, sections changed and a
summary. Export to `data/raw/srlc.csv` (columns `drug, application_number,
approval_date, section, summary`). This is the cleanest "label update date"
source; use DailyMed versions for pre-2016 and for non-safety edits.

## 4. FAERS — openFDA `drug/event` (open)

* Sample (count queries only): `python scripts/download_data.py --sample --start-quarter 2018Q1 --end-quarter 2024Q4`
  gives per-quarter drug × PT counts for the drug list plus margins, which feed
  `signal_scan.cumulative_from_counts`. Limitation: only the top-1000 PTs per
  drug-quarter and per quarter are returned by `count`.
* Full: `--faers-bulk --years 2004,...,2025` (quarterly JSON partitions, ~60 GB),
  flatten to (report_id, quarter, drug, pt) and de-duplicate by `safetyreportid`.
  Drug names: use `patient.drug[].openfda.generic_name` (suspect drugs only,
  `drugcharacterization == 1`, for the primary analysis).

## 5. FDA quarterly "Potential Signals of Serious Risks" (open, ground truth)

Index: https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers/potential-signals-serious-risksnew-safety-information-identified-fda-adverse-event-reporting-system
(one page per quarter since 2008; FDA renamed the system AEMS in 2026 — links may
move). `python scripts/download_data.py --potential-signals` fetches and parses
the tables (`label_lag.parse_potential_signals_html`); manual curation of the
`signal_text` → MedDRA PT mapping is expected (a few hundred rows).
Dhodapkar et al. (2022, *BMJ*) provide a curated 2008–2019 table of 603 signals
with resolution status in their supplement, useful for cross-checking.

## 6. MedDRA (licensed) and free alternatives

* MedDRA ASCII distribution (`MedAscii/pt.asc`, `llt.asc`, `hlt.asc`, ...):
  https://www.meddra.org/ — required for the full analysis
  (`term_extractor.MedDRADictionary.from_meddra_ascii`). Never commit these files.
* Development alternative: the bundled `SEED_TERMS` (~130 PTs) or a CSV of
  `pt,synonym` pairs built from SIDER 4.1 (http://sideeffects.embl.de/, MedDRA
  PT names with UMLS CUIs; check its licence) or the OHDSI vocabulary.
* Optional NER: `pip install scispacy` and the `en_ner_bc5cdr_md` model
  (https://allenai.github.io/scispacy/).

## Ethics

Public, de-identified sources. Label text and FAERS PTs contain no personal data;
FAERS narratives are not used. Do not send licensed MedDRA content to third-party
services in violation of the MSSO licence.
