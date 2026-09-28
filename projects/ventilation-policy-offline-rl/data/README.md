# Data acquisition

MIMIC-IV, eICU-CRD and HiRID are **credentialed** PhysioNet resources. The demo versions are open and
are what `scripts/download_data.py --sample` fetches so the pipeline can be exercised end-to-end.

Nothing under `data/` is committed.

## 0. Credentialing (once, ~2–4 weeks)

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training and upload the certificate.
3. Apply for credentialed access; then sign the DUA for each project:
   - https://physionet.org/content/mimiciv/3.1/
   - https://physionet.org/content/eicu-crd/2.0/
   - https://physionet.org/content/hirid/1.1.1/
4. Export credentials as environment variables (never commit them):
   ```bash
   export PHYSIONET_USER=your_username
   export PHYSIONET_PASSWORD='your_password'
   ```

## 1. Download

```bash
# open demo data (no credentials) – used by the --sample pipeline
python scripts/download_data.py --sample

# credentialed full data (reads PHYSIONET_USER / PHYSIONET_PASSWORD)
python scripts/download_data.py --mimic --eicu --hirid
```
Equivalent `wget` (what the script runs):
```bash
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASSWORD" \
     https://physionet.org/files/mimiciv/3.1/ -P data/raw/
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASSWORD" \
     https://physionet.org/files/eicu-crd/2.0/ -P data/raw/
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASSWORD" \
     https://physionet.org/files/hirid/1.1.1/ -P data/raw/
```
Google BigQuery access to `physionet-data.mimiciv_3_1_*` / `physionet-data.eicu_crd` is an alternative
for credentialed users; the SQL in `vent_rl.mdp_builder` is written for DuckDB but is standard enough
to port.

## 2. Expected layout

```
data/
  raw/
    physionet.org/files/mimiciv/3.1/hosp/*.csv.gz      # admissions, patients, omr, labevents, ...
    physionet.org/files/mimiciv/3.1/icu/*.csv.gz       # icustays, chartevents, d_items, ...
    physionet.org/files/eicu-crd/2.0/*.csv.gz          # patient, respiratoryCharting, respiratoryCare, vitalPeriodic, lab
    physionet.org/files/hirid/1.1.1/raw_stage/...      # observation_tables (parquet), general_table
    physionet.org/files/mimic-iv-demo/2.2/...          # --sample
    physionet.org/files/eicu-crd-demo/2.0.1/...        # --sample
  derived/
    mimic_derived.duckdb        # mimic-code derived concepts materialised (ventilator_setting, ventilation, bg, vitalsign, sofa)
    mimic_vent_mdp.parquet      # output of vent_rl.mdp_builder (one row per stay x 4h bin)
    eicu_vent_mdp.parquet
    hirid_vent_mdp.parquet
  reference/
    concept_map.csv             # harmonised variable definitions across sites (edit + version this file)
```

## 3. MIMIC-IV derived concepts

The community `mimic-code` repository provides the SQL for `ventilator_setting`, `ventilation`,
`oxygen_delivery`, `bg`, `vitalsign`, `sofa` etc.:

```bash
git clone https://github.com/MIT-LCP/mimic-code
# DuckDB build of MIMIC-IV + derived concepts:
#   mimic-code/mimic-iv/buildmimic/duckdb/import_duckdb.py  (see its README; --skip-indexes for speed)
#   mimic-code/mimic-iv/concepts_duckdb/  or  concepts_postgres/
```
`vent_rl.mdp_builder.MIMIC_VENT_SQL` assumes `mimiciv_derived.ventilator_setting`,
`mimiciv_derived.ventilation`, `mimiciv_derived.bg`, `mimiciv_derived.vitalsign`, `mimiciv_icu.icustays`,
`mimiciv_hosp.admissions`, `mimiciv_hosp.patients`, `mimiciv_hosp.omr` exist in the DuckDB file.

## 4. eICU-CRD

Key tables: `respiratorycharting` (`respchartvaluelabel` in {'Tidal Volume (set)', 'PEEP', 'FiO2',
'Plateau Pressure', 'Total RR', 'Mean Airway Pressure'}), `respiratorycare` (ventilator start/end offsets),
`vitalperiodic`, `lab`, `patient` (hospitalid, unitdischargestatus, hospitaldischargestatus, admissionheight,
admissionweight, gender, age). Offsets are minutes from unit admission.

## 5. HiRID

Use `hirid_variable_reference.csv` (in the release) to map `variableid` to concepts (PEEP, FiO2, tidal
volume, plateau pressure, PaO2, PaCO2, pH, SpO2 ...). Observations are 2-min resolution in
`raw_stage/observation_tables/parquet/`; aggregate to 4-h bins with `mdp_builder.bin_events`.
