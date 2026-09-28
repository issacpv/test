# Data acquisition: circadian-label-bias-icu

MIMIC-IV, MIMIC-IV-ED and eICU-CRD are **PhysioNet credentialed**. The demo subsets are open and sufficient to run the whole pipeline in `--sample` mode. Nothing under `data/` is committed.

## 0. Credentialing (once)

1. PhysioNet account: https://physionet.org/register/
2. CITI "Data or Specimens Only Research" training: https://physionet.org/settings/training/
3. Sign the DUA for each database: MIMIC-IV v3.1 (https://physionet.org/content/mimiciv/3.1/), eICU-CRD v2.0 (https://physionet.org/content/eicu-crd/2.0/), optionally MIMIC-IV-ED v2.2 (https://physionet.org/content/mimic-iv-ed/2.2/).
4. Export credentials in the shell only:
   ```bash
   export PHYSIONET_USER="your_username"
   export PHYSIONET_PASS="your_password"
   ```

## 1. Open demos

```bash
python scripts/download_data.py --sample
```
fetches the MIMIC-IV Clinical Database Demo v2.2 and the eICU-CRD Demo v2.0.1 over plain HTTPS.

## 2. Full databases (credentialed)

```bash
python scripts/download_data.py --mimic-iv           # only the tables below (~25 GB)
python scripts/download_data.py --eicu
python scripts/download_data.py --mimic-iv-ed        # optional
```
Manual equivalent: `wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/mimiciv/3.1/ -P data/`.

Tables used and the timestamp columns that matter:

| Database | Table | Columns |
|---|---|---|
| MIMIC-IV | `hosp/admissions` | `admittime`, `dischtime`, `deathtime`, `admission_location`, `edregtime` |
| MIMIC-IV | `hosp/patients` | `anchor_age`, `dod` |
| MIMIC-IV | `hosp/transfers` | `intime`, `outtime`, `careunit` |
| MIMIC-IV | `icu/icustays` | `intime`, `outtime`, `first_careunit` |
| MIMIC-IV | `icu/chartevents` | `charttime`, `storetime`, `itemid`, `value`, `warning` |
| MIMIC-IV | `hosp/labevents` | `charttime`, `storetime`, `itemid`, `flag`, `valuenum` |
| MIMIC-IV | `icu/inputevents`, `icu/procedureevents` | `starttime`, `endtime`, `storetime`, `itemid` |
| MIMIC-IV | `hosp/prescriptions`, `hosp/emar` | `starttime`, `charttime`, `scheduletime` |
| MIMIC-IV | `hosp/microbiologyevents` | `charttime`, `chartdate`, `storetime` |
| eICU | `patient` | `unitadmittime24`, `hospitaladmittime24`, `unitdischargeoffset`, `hospitaldischargeoffset`, `unitdischargestatus`, `hospitalid` |
| eICU | `hospital` | `numbedscategory`, `teachingstatus`, `region` |
| eICU | `lab`, `nurseCharting`, `vitalPeriodic`, `infusionDrug`, `medication`, `microLab`, `treatment`, `respiratoryCare` | `*offset` columns (minutes from unit admission) plus `labresultrevisedoffset` (documentation delay proxy) |

## 3. Build DuckDB and event tables

```bash
python scripts/download_data.py --build-duckdb --db data/mimiciv_demo.duckdb --source data/mimic-iv-demo/2.2
python scripts/download_data.py --build-duckdb --db data/eicu_demo.duckdb --source data/eicu-crd-demo/2.0.1
```
The `circadian_bias.label_timing.mimic_event_table()` helper documents the SQL used to derive one row per (stay, event_type) with `event_time`, `clock_hour`, `weekday`, `store_delay_min`.

## 4. Label definitions

- YAIB task definitions (mortality 48 h, LOS, AKI, sepsis): https://github.com/rvandewater/YAIB
- mimic-code concepts: `sepsis3.sql`, `kdigo_stages.sql`, `ventilation.sql`, `vasoactive_agent.sql`, `code_status.sql` (for withdrawal-of-care sensitivity) fetched by `--sample` into `data/reference/`.

## Expected layout

```
data/
  mimic-iv-demo/2.2/{hosp,icu}/*.csv.gz
  eicu-crd-demo/2.0.1/*.csv.gz
  mimiciv/3.1/{hosp,icu}/*.csv.gz         # credentialed
  eicu/2.0/*.csv.gz                        # credentialed
  mimic-iv-ed/2.2/ed/*.csv.gz              # credentialed, optional
  reference/*.sql
  *.duckdb                                 # local only
  events/                                  # derived event tables (parquet), never committed
```
