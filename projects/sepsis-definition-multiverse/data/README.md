# Data acquisition: sepsis-definition-multiverse

MIMIC-IV and eICU-CRD are **PhysioNet credentialed** resources. The demo subsets are open and are enough to develop and test the whole pipeline. Nothing under `data/` is ever committed.

## 0. Credentialing (once)

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training and upload the certificate: https://physionet.org/settings/training/
3. Apply for credentialed access and sign the DUA for each database:
   - MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
   - eICU-CRD v2.0: https://physionet.org/content/eicu-crd/2.0/
4. Export your credentials in the shell (never in code or git):
   ```bash
   export PHYSIONET_USER="your_username"
   export PHYSIONET_PASS="your_password"
   ```

## 1. Open demo subsets (no credentialing)

```bash
python scripts/download_data.py --sample
```
downloads MIMIC-IV Clinical Database Demo v2.2 (https://physionet.org/content/mimic-iv-demo/2.2/) and eICU-CRD Demo v2.0.1 (https://physionet.org/content/eicu-crd-demo/2.0.1/) with plain `wget`-style HTTP requests, plus the mimic-code Sepsis-3 SQL for reference.

## 2. Full databases (credentialed)

```bash
python scripts/download_data.py --mimic-iv          # ~30 GB of csv.gz into data/mimiciv/3.1/
python scripts/download_data.py --eicu              # ~20 GB into data/eicu/2.0/
```
Equivalent manual commands:
```bash
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/mimiciv/3.1/ -P data/
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/eicu-crd/2.0/ -P data/
```
Only the tables used are needed: MIMIC-IV `hosp/{admissions,patients,labevents,d_labitems,prescriptions,emar,emar_detail,microbiologyevents}` and `icu/{icustays,chartevents,d_items,inputevents,outputevents,procedureevents}`; eICU `{patient,lab,vitalPeriodic,vitalAperiodic,infusionDrug,medication,microLab,treatment,respiratoryCare,nurseCharting,intakeOutput,apacheApsVar}`.

## 3. Build a DuckDB

```bash
python scripts/download_data.py --build-duckdb --db data/mimiciv_demo.duckdb --source data/mimic-iv-demo/2.2
```
creates one table per CSV (`hosp_*`, `icu_*`) with `read_csv_auto`. The same command with `--source data/mimiciv/3.1` builds the full database (allow ~1 h and ~60 GB of disk).

## 4. Reference implementations (open)

- mimic-code: `mimic-iv/concepts/sepsis/sepsis3.sql`, `suspicion_of_infection.sql`, `score/sofa.sql` (fetched by `--sample` into `data/reference/`).
- ricu concept dictionary: https://github.com/eth-mds/ricu (R); YAIB: https://github.com/rvandewater/YAIB.
- Seymour et al. (2016 JAMA) supplementary definitions are the primary source for windows.

## Expected layout

```
data/
  mimic-iv-demo/2.2/{hosp,icu}/*.csv.gz
  eicu-crd-demo/2.0.1/*.csv.gz
  mimiciv/3.1/{hosp,icu}/*.csv.gz          # credentialed
  eicu/2.0/*.csv.gz                          # credentialed
  reference/{sepsis3.sql,suspicion_of_infection.sql,sofa.sql}
  *.duckdb                                   # built locally, never committed
  hourly/                                    # long hourly concept tables (parquet), never committed
```
