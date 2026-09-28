# Data acquisition

Nothing in this folder is committed. MIMIC-IV and eICU are credentialed; Synthea and the PhysioNet
demo databases are open.

## 1. Synthea (open, fully synthetic)

Option A: prebuilt sample export (1,000-ish patients, CSV):

```bash
cd projects/synthetic-ehr-transportability
python scripts/download_data.py --synthea-sample --out data/synthea
```

Option B: generate your own cohort (requires Java 11+; ~1 h for 100k patients):

```bash
python scripts/download_data.py --synthea-generate 20000 --seed 1 --out data/synthea
# runs: java -jar synthea-with-dependencies.jar -p 20000 -s 1 --exporter.csv.export=true --exporter.fhir.export=false
```

Expected files: `data/synthea/csv/{patients,encounters,conditions,observations,medications,procedures}.csv`.
Record the Synthea release/commit and the seed in `data/synthea/VERSION.txt` (the script does this).

## 2. PhysioNet demo databases (open; for pipeline development)

```bash
python scripts/download_data.py --mimic-demo --out data/mimic-iv-demo     # ~100 patients
python scripts/download_data.py --eicu-demo  --out data/eicu-demo         # ~2,500 stays
```

Both use `wget -r -N -c -np` on `https://physionet.org/files/mimic-iv-demo/2.2/` and
`https://physionet.org/files/eicu-crd-demo/2.0.1/`.

## 3. MIMIC-IV v3.1 and eICU-CRD v2.0 (credentialed)

1. Complete CITI "Data or Specimens Only Research" training and become a credentialed PhysioNet user.
2. Sign the DUA for each project on PhysioNet.
3. Export credentials in your shell (never commit them):

```bash
export PHYSIONET_USERNAME=you
export PHYSIONET_PASSWORD=...
python scripts/download_data.py --mimic-full --out data/mimiciv --run   # prints the wget command without --run
python scripts/download_data.py --eicu-full  --out data/eicu --run
```

The commands are:

```
wget -r -N -c -np --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" https://physionet.org/files/mimiciv/3.1/
wget -r -N -c -np --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" https://physionet.org/files/eicu-crd/2.0/
```

Tables used: MIMIC-IV `hosp/{patients,admissions,diagnoses_icd,labevents,d_labitems}`, `icu/icustays`;
eICU `{patient,diagnosis,pastHistory,lab,apachePatientResult}`.

Load into DuckDB (fast, out-of-core):

```python
import duckdb
con = duckdb.connect("data/mimiciv.duckdb")
con.execute("CREATE TABLE admissions AS SELECT * FROM read_csv_auto('data/mimiciv/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz')")
```

The encounter-level feature SQL for MIMIC-IV and eICU is documented in `src/synth_transport/ontology.py`.

## 4. Learned generators

```bash
pip install sdv            # CTGAN, TVAE, GaussianCopula
pip install synthcity      # DDPM and others (large dependency set; use a separate venv)
```

## Expected layout

```
data/
  synthea/csv/*.csv
  synthea/VERSION.txt
  mimic-iv-demo/physionet.org/files/mimic-iv-demo/2.2/...
  eicu-demo/physionet.org/files/eicu-crd-demo/2.0.1/...
  mimiciv/physionet.org/files/mimiciv/3.1/{hosp,icu}/*.csv.gz
  eicu/physionet.org/files/eicu-crd/2.0/*.csv.gz
  features/{mimic_train,mimic_holdout,eicu,synthea}.parquet
  synthetic/<generator>_seed<k>.parquet
outputs/
```
