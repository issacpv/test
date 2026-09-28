# Data acquisition — device-recall-prediction

Everything is open; nothing under `data/` except this README is committed.

## Layout

```
data/
├── README.md
├── raw/
│   ├── openfda_sample/            # --sample: classification, k510, pma, recalls, enforcement,
│   │                              #   recalls_with_class, maude_sample, maude_monthly_counts, udi_sample
│   ├── bulk/<endpoint>/*.json.zip # --bulk partitions from https://api.fda.gov/download.json
│   ├── 510k_summaries/K######.pdf # --summaries
│   ├── k_numbers.txt              # your list of K numbers for --summaries
│   └── fda_ai_enabled_devices.xlsx (or .csv)   # --ai-list or manual download
└── processed/                     # parquet tables built by the pipeline (ignored)
```

## 1. openFDA device endpoints (open; optional free API key)

Key: https://open.fda.gov/apis/authentication/ → `export OPENFDA_API_KEY=...`
(240 req/min, 120 000/day; without a key 40 req/min, 1 000/day).

| Endpoint | Docs | Used fields |
|---|---|---|
| `device/event` (MAUDE) | https://open.fda.gov/apis/device/event/ | `mdr_report_key, date_received, date_of_event, event_type, report_source_code, product_problems, device[].brand_name/generic_name/device_report_product_code/manufacturer_d_name/model_number/udi_di, mdr_text[].text (text_type_code "Description of Event or Problem"), patient[].sequence_number_outcome` |
| `device/recall` | https://open.fda.gov/apis/device/recall/ | `res_event_number, product_res_number, product_code, k_numbers[], pma_numbers[], recalling_firm, root_cause_description, event_date_initiated, product_description` |
| `device/enforcement` | https://open.fda.gov/apis/device/enforcement/ | `event_id (== recall.res_event_number), classification (Class I/II/III), recall_initiation_date, center_classification_date, voluntary_mandated` |
| `device/510k` | https://open.fda.gov/apis/device/510k/ | `k_number, applicant, device_name, product_code, decision_code, decision_date, clearance_type, statement_or_summary, third_party_flag, openfda.device_class` |
| `device/pma` | https://open.fda.gov/apis/device/pma/ | `pma_number, supplement_number, supplement_type, applicant, trade_name, product_code, decision_date` |
| `device/classification` | https://open.fda.gov/apis/device/classification/ | `product_code, device_name, device_class, regulation_number, medical_specialty, review_panel, implant_flag, life_sustain_support_flag, submission_type_id` |
| `device/udi` (GUDID) | https://open.fda.gov/apis/device/udi/ | `identifiers[].id, brand_name, company_name, product_codes[].code, premarket_submissions[].submission_number` |

Quick sample (≈ 10–20 min with a key):

```bash
python scripts/download_data.py --sample --product-codes LZG --max-maude 500 --start 2015-01-01 --end 2025-06-30
```

Full database: MAUDE alone is > 15 M reports (≈ 20 GB zipped JSON); the other
endpoints are small (< 1 GB total).

```bash
python scripts/download_data.py --bulk --endpoints 510k,recall,enforcement,classification,pma,udi
python scripts/download_data.py --bulk --endpoints event      # MAUDE, large
```

Bulk partitions are JSON objects with a `results` list; flatten with
`device_recall.openfda_device.flatten_*`. Note that MAUDE reports have follow-up
versions and manufacturer/user-facility duplicates of the same event; de-duplicate
by `mdr_report_key` and, for event counting, by (`report_number` prefix, `date_of_event`,
device) as recommended in Ensign & Cohen (2017, *eGEMs*).

## 2. 510(k) summaries → predicates (open)

openFDA does not expose predicate devices. For each K number the public summary
PDF sits at `https://www.accessdata.fda.gov/cdrh_docs/pdf<YY>/K<number>.pdf`
(`device_recall.predicate_graph.summary_pdf_url`) or is linked from the 510(k)
database page `https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K<number>`.
Only submissions with `statement_or_summary == "Summary"` have a summary
(statements do not list predicates); summaries are text-searchable PDFs for
most submissions after ~2000, scanned images before (OCR with `ocrmypdf` if needed).

```bash
# k_numbers.txt: one K number per line (e.g. from data/raw/openfda_sample/k510.csv)
python scripts/download_data.py --summaries --k-file data/raw/k_numbers.txt
```

Be polite: ~1 request/s (the script sleeps 1 s). For ~40 000 summaries this is a
multi-day background job; run it per product-code cohort.

## 3. FDA list of AI-enabled medical devices (open)

https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices
— downloadable as CSV/XLSX (columns: Date of Final Decision, Submission Number,
Device, Company, Panel (lead), Primary Product Code). `--ai-list` tries to fetch
the linked file; otherwise download manually to `data/raw/fda_ai_enabled_devices.xlsx`.
Join on `Submission Number` = `k_number` / `pma_number` / DEN number.

## 4. Optional enrichment

* FDA 510(k) "Releasable" database (weekly zip, includes all K numbers since 1976):
  https://www.fda.gov/medical-devices/510k-clearances/downloadable-510k-files
* FDA Recalls database (CDRH) for `Recall Class` cross-check:
  https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
* FDA TPLC (Total Product Life Cycle) database, per product code MDR and recall
  summaries: https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfTPLC/tplc.cfm

## Ethics

MAUDE narratives are de-identified but occasionally contain incidental patient
details; do not attempt re-identification and do not upload narratives to
third-party LLM APIs without institutional approval. Never commit data.
