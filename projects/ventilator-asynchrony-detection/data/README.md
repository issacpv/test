# Data acquisition

Nothing under `data/` is committed except this file. The charted ICU databases (HiRID, eICU-CRD,
MIMIC-IV) are PhysioNet **credentialed**; simulated waveforms are generated locally (open); the
open waveform sources listed in section 3 must each be checked for current availability and licence.

Expected layout:

```
data/
  simulated/                       # scripts/download_data.py --simulate: npz per scenario with flow, paw, labels
  hirid/1.1.1/
    reference_data/hirid_variable_reference.csv  general_table.csv
    raw_stage/observation_tables/parquet/part-*.parquet      (from observation_tables_parquet.tar.gz)
    raw_stage/pharma_records/parquet/...                      (sedation covariates)
  eicu/2.0/
    patient.csv.gz  apachePatientResult.csv.gz  respiratoryCharting.csv.gz  respiratoryCare.csv.gz
    vitalPeriodic.csv.gz (monitor respiration at 5 min)  infusionDrug.csv.gz (sedation)
  mimiciv/3.1/icu/
    d_items.csv.gz  icustays.csv.gz  chartevents.csv.gz  procedureevents.csv.gz  inputevents.csv.gz
  waveforms/                       # open ventilator-waveform datasets with breath-level labels (see section 3)
  kaggle-ventilator/               # optional: Google Brain "Ventilator Pressure Prediction" (artificial lung; segmentation pre-training only)
```

## 1. Simulated waveforms (open)

```bash
python scripts/download_data.py --simulate --out data/simulated --duration 600 --seeds 0 1 2
```

writes one `.npz` per scenario (`controlled`, `synchronous_assisted`, `ineffective_efforts`,
`double_triggering`, `reverse_triggering`, `auto_triggering`) and seed with `t, flow, paw, volume,
pmus`, the breath table (`start_idx, insp_end_idx, trigger, label`) and the effort table.

## 2. Charted ICU databases (PhysioNet credentialed)

```bash
export PHYSIONET_USERNAME=your_user
export PHYSIONET_PASSWORD='your_password'
python scripts/download_data.py --dataset hirid --out data/hirid/1.1.1 --sample        # reference tables only
python scripts/download_data.py --dataset hirid --out data/hirid/1.1.1                 # + raw observation tables (~ tens of GB)
python scripts/download_data.py --dataset eicu --out data/eicu/2.0
python scripts/download_data.py --dataset mimiciv-icu --out data/mimiciv/3.1/icu
```

- HiRID (https://physionet.org/content/hirid/1.1.1/): ventilator and monitor variables at 2-min
  resolution in the raw observation tables; find the variable ids with
  `pva_detect.cohort.lookup_variables(reference, HIRID_HINTS, "Variable Name", "ID")` and confirm by hand.
- eICU-CRD (https://physionet.org/content/eicu-crd/2.0/): `respiratoryCharting.respchartvaluelabel`
  holds ventilator settings/observations (labels vary by hospital; use `EICU_HINTS` and verify),
  `vitalPeriodic.respiration` is the bedside-monitor rate at 5 min.
- MIMIC-IV (https://physionet.org/content/mimiciv/3.1/): `chartevents` itemids in
  `pva_detect.cohort.MIMIC_VENT_ITEMS` (RR set 224688, RR spontaneous 224689, RR total 224690, VT observed
  224685, minute volume 224687, peak pressure 224695, PEEP 220339, mode 223849) and the monitor
  respiratory rate 220210; verify every id against `d_items`.

## 3. Open ventilator waveform datasets with asynchrony labels (verify availability)

Candidates to check at the start of the project (none is downloaded automatically; record the
version and licence you obtain):

- Breath-level annotated datasets released with recent PVA papers (a 2026 Scientific Reports paper on
  1-D U-Net segmentation of ventilator waveforms, 33 patients / 9,719 breaths; the 2025 PVADet study;
  the 2025 real-time circuit-event study with 3.1 million breaths). Check each paper's data-availability
  statement and the associated repositories.
- UC Davis APL (Annotation Platform for Lung) derived releases and the `ventmap` tooling (GitHub).
- Simulated/bench datasets accompanying Bakkes et al. (Comput Methods Programs Biomed 2023).
- Google Brain "Ventilator Pressure Prediction" (Kaggle, open): artificial-lung pressure/flow at
  fixed settings, no patient efforts; useful only for segmentation pre-training.
  `kaggle competitions download -c ventilator-pressure-prediction -p data/kaggle-ventilator`
- VitalDB (open): some cases carry anaesthesia-machine airway-pressure/CO2 waveform tracks; check
  the track list (`https://api.vitaldb.net/trks`) for `Primus/*` waveform names.

## 4. Verify

```bash
python scripts/download_data.py --verify --out data
```
