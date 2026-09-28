# Data acquisition

All sources are open. Nothing under `data/` is committed except this file.

## 0. Optional API key

openFDA allows 40 requests/min and 1,000/day without a key, 240/min and 120,000/day with a free key
(https://open.fda.gov/apis/authentication/). Export it in the shell only:

```bash
export OPENFDA_API_KEY="..."
```

## 1. FDA AI-enabled device list (manual step)

The FDA publishes the list as a downloadable table on
https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices
(the export link changes between updates). Save it as `data/raw/fda_ai_devices.xlsx` (or `.csv`). Expected columns
(names are matched loosely by `ai_device_taxonomy.linkage.load_ai_device_list`): date of final decision,
submission number, device name, company, panel (lead), primary product code. If you have the direct link:

```bash
python scripts/download_data.py --ai-list-url "<direct download URL>"
```

## 2. openFDA sample (smoke test, ~minutes)

```bash
python scripts/download_data.py --sample
```

Pulls up to 300 MAUDE reports for each of a few product codes that contain AI-enabled devices
(`--product-codes`, default `QAS,QFM,POK,LLZ`), plus the `device/classification` rows for those codes and the
`device/510k` records of the submission numbers in the AI list when it is present.

## 3. Full openFDA download

```bash
python scripts/download_data.py --full                  # MAUDE bulk partitions + 510k + pma + classification + recall
python scripts/download_data.py --full --endpoints event # only the MAUDE bulk files
```

MAUDE bulk files come from the openFDA manifest `https://api.fda.gov/download.json`
(`results.device.event.partitions[*].file`); each partition is a zip containing one JSON file. Expect ~20 GB.
Paged API queries use the `Link: <...>; rel="next"` header (`search_after` cursor) because `skip` is capped at
25,000.

Manual equivalents:

```bash
curl -s "https://api.fda.gov/download.json" | python -c "import json,sys; m=json.load(sys.stdin); print('\n'.join(p['file'] for p in m['results']['device']['event']['partitions']))"
curl -s "https://api.fda.gov/device/event.json?search=device.device_report_product_code:QAS&limit=100"
curl -s "https://api.fda.gov/device/510k.json?search=k_number:K123456"
```

## 4. Expected layout

```
data/
  README.md
  annotation_guideline.md          # taxonomy definitions and annotation rules (versioned)
  raw/
    fda_ai_devices.xlsx            # manual download
    openfda/
      device_event/*.json.zip      # bulk partitions (full mode)
      sample/event_<CODE>.jsonl    # sample mode
      classification.jsonl, k510.jsonl, pma.jsonl, recall.jsonl, udi_<CODE>.jsonl
  derived/
    ai_devices.parquet             # cleaned AI list with product codes and decision dates
    maude_flat.parquet             # flattened MDRs (one row per report)
    linkage.parquet                # (report_number, submission_number, tier, brand_score, company_score)
    labels.parquet                 # taxonomy labels with evidence spans
    device_years.parquet           # denominators for AI and comparator devices
```
