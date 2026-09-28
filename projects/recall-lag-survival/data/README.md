# Data acquisition

All sources are open FDA data via openFDA. Nothing under `data/` is committed except this file and, later,
`data/firm_aliases.csv` (a hand-curated table of firm-name aliases; no personal data).

## 0. Optional API key

```bash
export OPENFDA_API_KEY="..."   # 240 req/min & 120k/day with a key; 40/min & 1k/day without
```

## 1. Sample (smoke test)

```bash
python scripts/download_data.py --sample
```

Fetches 500 recent `device/recall` records, their `device/enforcement` counterparts (by `res_event_number` /
`product_res_number`), and up to 200 MAUDE reports for each of the 20 most frequent product codes among them.
Writes JSON lines under `data/raw/openfda/sample/`.

## 2. Full download

```bash
python scripts/download_data.py --full                          # everything below
python scripts/download_data.py --full --endpoints recall,enforcement,510k,pma,classification
python scripts/download_data.py --full --endpoints event        # MAUDE bulk partitions only (~20 GB)
```

* `device/event` is fetched as bulk zip partitions listed in `https://api.fda.gov/download.json`
  (`results.device.event.partitions[*].file`).
* `device/recall`, `device/enforcement`, `device/510k`, `device/pma`, `device/classification` are paged with the
  `Link: <...>; rel="next"` header (the `skip` parameter is capped at 25,000).
* `device/registrationlisting` (firm-size proxy) is large; fetch it with `--endpoints registrationlisting` only
  when needed.

Manual equivalents:

```bash
curl -s "https://api.fda.gov/device/recall.json?limit=100&sort=event_date_initiated:desc"
curl -s "https://api.fda.gov/device/enforcement.json?search=classification:%22Class+I%22&limit=100"
curl -s "https://api.fda.gov/download.json" | python -c "import json,sys; m=json.load(sys.stdin); print(len(m['results']['device']['event']['partitions']))"
```

## 3. Expected layout

```
data/
  README.md
  firm_aliases.csv                     # (alias, canonical_firm) hand-curated
  raw/openfda/
    device_event/*.json.zip            # bulk MAUDE partitions
    recall.jsonl, enforcement.jsonl, k510.jsonl, pma.jsonl, classification.jsonl
    sample/...
  derived/
    reports_min.parquet                # (report_number, date_received, date_of_event, event_type, product_code, firm_norm, key)
    monthly_counts.parquet             # (key, month, n_all, n_serious)
    recalls.parquet                    # (res_event_number, key, initiated, posted, classified, class, root_cause, pathway)
    tte.parquet                        # time-to-event table (key, origin, entry_years, time_years, event, covariates)
```

## 4. Key date fields

| Table | Field | Meaning |
|---|---|---|
| event | `date_of_event` | when the event occurred (often missing) |
| event | `date_received` | when FDA received the report |
| event | `date_report` | manufacturer report date |
| recall | `event_date_initiated` | firm-initiated recall start |
| recall | `event_date_posted` | posting on FDA's recall database |
| enforcement | `recall_initiation_date`, `center_classification_date`, `report_date` | initiation, FDA classification, enforcement report |
