# Data acquisition

All four databases contain de-identified but sensitive patient data. None of it is stored in this
repository. `data/` is git-ignored except for this file.

## 0. Credentials

PhysioNet (MIMIC-IV, eICU-CRD, HiRID):

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training and upload the certificate:
   https://physionet.org/settings/credentialing/
3. Sign the data use agreement (DUA) on each project page (links below). Approval takes days to weeks.
4. Export credentials in the shell (never write them into files that could be committed):

```bash
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"
```

AmsterdamUMCdb is **not** on PhysioNet. Register and sign the end-user licence at
https://amsterdammedicaldatascience.nl/amsterdamumcdb/ ; a download link is emailed after approval.

## 1. Download

`scripts/download_data.py` wraps `wget` with `--user/--password` from the environment and supports
`--sample` (only the small dictionary/demographic tables) for smoke tests.

```bash
python scripts/download_data.py --db mimiciv --sample          # dictionaries + patients only
python scripts/download_data.py --db mimiciv                   # full hosp + icu modules
python scripts/download_data.py --db eicu
python scripts/download_data.py --db hirid
python scripts/download_data.py --db aumcdb --aumcdb-url "<link from the licence email>"
```

Equivalent manual commands:

```bash
# MIMIC-IV v3.1 (~30 GB compressed)
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
     https://physionet.org/files/mimiciv/3.1/ -P data/raw/
# eICU-CRD v2.0
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
     https://physionet.org/files/eicu-crd/2.0/ -P data/raw/
# HiRID v1.1.1 (raw_stage is what the ontology uses; the pre-processed stages are optional)
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
     https://physionet.org/files/hirid/1.1.1/ -P data/raw/
```

HiRID ships as tarballs (`raw_stage/observation_tables_csv.tar.gz`, `pharma_records_csv.tar.gz`,
`reference_data.tar.gz`); extract them in place. AmsterdamUMCdb ships as a set of large CSVs
(`numericitems.csv` is ~50 GB); keep them compressed with `gzip` and let DuckDB read the `.gz`.

## 2. Expected layout

```
data/
  raw/
    physionet.org/files/mimiciv/3.1/hosp/{patients,admissions,labevents,d_labitems,prescriptions,emar,microbiologyevents,diagnoses_icd}.csv.gz
    physionet.org/files/mimiciv/3.1/icu/{icustays,chartevents,inputevents,outputevents,d_items,procedureevents}.csv.gz
    physionet.org/files/eicu-crd/2.0/{patient,vitalPeriodic,vitalAperiodic,lab,nurseCharting,infusionDrug,medication,intakeOutput,microLab,hospital}.csv.gz
    physionet.org/files/hirid/1.1.1/raw_stage/observation_tables/csv/part-*.csv
    physionet.org/files/hirid/1.1.1/raw_stage/pharma_records/csv/part-*.csv
    physionet.org/files/hirid/1.1.1/reference_data/{general_table.csv,hirid_variable_reference.csv,pharma_reference.csv}
    aumcdb/1.0.2/{admissions,numericitems,listitems,drugitems,freetextitems,procedureorderitems,processitems}.csv.gz
  interim/      # long-format concept tables written by icu_transport.cohorts (parquet)
  processed/    # hourly feature matrices + labels per database (parquet)
```

Point the code at the roots with environment variables or the `roots` argument:

```bash
export ICU_MIMICIV_ROOT=data/raw/physionet.org/files/mimiciv/3.1
export ICU_EICU_ROOT=data/raw/physionet.org/files/eicu-crd/2.0
export ICU_HIRID_ROOT=data/raw/physionet.org/files/hirid/1.1.1
export ICU_AUMCDB_ROOT=data/raw/aumcdb/1.0.2
```

## 3. Verify the ontology before extracting anything

```python
import duckdb
from icu_transport.cohorts import verify_ontology
con = duckdb.connect()
report = verify_ontology(con, "mimiciv", root="data/raw/physionet.org/files/mimiciv/3.1")
print(report[report.status != "ok"])
```

Concept ids for HiRID and AmsterdamUMCdb are marked `verified=False` in `cohorts.py` until this check
has been run on the real dictionaries; cross-check against the ricu concept dictionary
(`ricu::load_dictionary()`) and YAIB's `configs/` when in doubt.

## 4. Optional: YAIB cohorts for cross-checks

```bash
pip install yaib   # or: git clone https://github.com/rvandewater/YAIB
```

YAIB's cohort extraction is R-based (ricu). Use it to generate the reference cohorts and compare
prevalence per task and database with the numbers produced here (target: within +-2%).
