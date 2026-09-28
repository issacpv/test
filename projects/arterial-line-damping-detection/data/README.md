# Data acquisition

Nothing under `data/` is committed except this file. VitalDB is free/open (registration on the
website, CC BY-NC-SA); the MIMIC resources are PhysioNet **credentialed**. Synthetic records with
embedded flush tests are generated locally and are the only fully open input.

Expected layout:

```
data/
  synthetic/                        # scripts/download_data.py --simulate: npz records with known (fn, zeta) and flush events
  vitaldb/
    cases.csv  trks.csv
    tracks/<caseid>/SNUADC_ART.csv  Solar8000_NIBP_SBP.csv  Solar8000_NIBP_DBP.csv  Solar8000_NIBP_MBP.csv
  mimic3wdb-matched/1.0/            # MIMIC-III Waveform Database Matched Subset (ABP 125 Hz + numerics)
    RECORDS  RECORDS-waveforms  RECORDS-numerics
    p00/p000020/p000020-2183-04-28-17-47.hea/.dat  ...  (+ *n.hea numerics with ABP Sys/Dias/Mean, NBP)
  mimiciii/1.4/
    D_ITEMS.csv.gz  CHARTEVENTS.csv.gz  INPUTEVENTS_MV.csv.gz  PROCEDUREEVENTS_MV.csv.gz  ICUSTAYS.csv.gz
  mimic4wdb/0.1.0/                  # MIMIC-IV Waveform v0.1.0 (200 records) - pilot for MIMIC-IV linkage
    RECORDS  RECORDS-waveforms  waves/...
  mimiciv/3.1/icu/
    d_items.csv.gz  chartevents.csv.gz  inputevents.csv.gz  procedureevents.csv.gz  icustays.csv.gz
```

## 1. Synthetic records (open)

```bash
python scripts/download_data.py --simulate --out data/synthetic --n-records 20
```

Each `.npz` holds a true Windkessel ABP passed through a random catheter system (fn, zeta), with
two fast-flush tests embedded, plus the ground truth; use it to test the whole pipeline end to end.

## 2. VitalDB (open; 500 Hz arterial waveform + NIBP)

```bash
python scripts/download_data.py --dataset vitaldb --out data/vitaldb            # cases.csv + trks.csv
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks --sample   # 5 cases with ART + NIBP
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks --max-cases 500
```

Flush tests during surgery are visible as 300 mmHg square waves in `SNUADC/ART`. Citation: Lee H-C. et al., Sci Data 2022.

## 3. MIMIC-III Waveform Database Matched Subset (credentialed)

```bash
export PHYSIONET_USERNAME=your_user
export PHYSIONET_PASSWORD='your_password'
python scripts/download_data.py --dataset mimic3wdb-matched --out data/mimic3wdb-matched/1.0 --sample
```

`--sample` fetches the `RECORDS*` lists and the first patient folder. For the cohort, filter
`RECORDS-waveforms` to records whose `.hea` lists `ABP`, then mirror those folders:

```bash
wget -r -N -c -np -nH --cut-dirs=2 --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" \
  -P data/mimic3wdb-matched https://physionet.org/files/mimic3wdb-matched/1.0/p00/p000020/
```

The `*n` numerics records carry the monitor's own ABP Sys/Dias/Mean and NBP values at 1 Hz / per reading,
which give an independent check of the beat-detected values and the NIBP pairs.

## 4. MIMIC-III / MIMIC-IV clinical tables (credentialed)

```bash
python scripts/download_data.py --dataset mimiciii --out data/mimiciii/1.4 --sample     # D_ITEMS, ICUSTAYS
python scripts/download_data.py --dataset mimiciii --out data/mimiciii/1.4              # + CHARTEVENTS, INPUTEVENTS_MV, PROCEDUREEVENTS_MV
python scripts/download_data.py --dataset mimiciv-icu --out data/mimiciv/3.1/icu
```

Use `D_ITEMS`/`d_items` to locate: NIBP systolic/diastolic/mean (MIMIC-IV 220179/220180/220181), arterial
BP charted values (220050/220051/220052), vasopressor infusions (`inputevents`, e.g. norepinephrine 221906),
arterial-line insertion/removal (`procedureevents`, label contains "Arterial Line") and line site.
Verify every itemid against the dictionary before use.

## 5. MIMIC-IV Waveform v0.1.0 (credentialed, pilot)

```bash
python scripts/download_data.py --dataset mimic4wdb --out data/mimic4wdb/0.1.0 --sample
```

## 6. Verify

```bash
python scripts/download_data.py --verify --out data
```
