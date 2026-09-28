# Data acquisition

Nothing under `data/` is committed except this file. VitalDB is free/open (registration on the
website, CC BY-NC-SA licence); the MIMIC resources are PhysioNet **credentialed**.

Expected layout:

```
data/
  vitaldb/
    cases.csv                     # https://api.vitaldb.net/cases  (clinical info per case)
    trks.csv                      # https://api.vitaldb.net/trks   (track index: caseid, tname, tid)
    tracks/<caseid>/<track>.csv   # downloaded numeric / waveform tracks (Time, value)
  mimic3wdb-matched/1.0/          # MIMIC-III Waveform Database Matched Subset (ABP at 125 Hz)
    p00/p000020/p000020-2183-04-28-17-47.hea ...  (+ .dat, and the *n.hea numerics)
  mimiciii/1.4/
    NOTEEVENTS.csv.gz             # echo reports: CATEGORY = 'Echo' (semi-structured measurement lines)
    CHARTEVENTS.csv.gz  D_ITEMS.csv.gz   # thermodilution / CCO cardiac output (find itemids by label)
  mimic4wdb/0.1.0/                # MIMIC-IV Waveform Database v0.1.0 (200 records / 198 patients; ABP, ECG, PPG)
  mimic-iv-echo/                  # MIMIC-IV-ECHO (v0.1: 7,243 studies / 4,579 patients, 2017-2019; DICOM) and,
                                  # in later versions, cardiologist-recorded measurement tables
  mimiciv/3.1/icu/                # chartevents.csv.gz + d_items.csv.gz for PA-catheter cardiac output
```

## 1. VitalDB (open; ~6,300 surgical cases)

Register at https://vitaldb.net (accept the data-use terms), then:

```bash
python scripts/download_data.py --dataset vitaldb --out data/vitaldb                # cases.csv + trks.csv
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks       # all cases with CardioQ or EV1000 SV + ART
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks --sample   # first 5 eligible cases
```

Tracks used: `SNUADC/ART` (arterial waveform, 500 Hz), `CardioQ/SV`, `CardioQ/CO`, `CardioQ/FTc`
(oesophageal Doppler, independent of the arterial waveform), `EV1000/SV`, `EV1000/CO`, `EV1000/SVV`
and `Vigileo/SV`, `Vigileo/CO` (FloTrac pulse-contour devices, the commercial comparator),
`Solar8000/NIBP_SBP`/`_DBP`. The `vitaldb` package (`pip install vitaldb`) gives
`vitaldb.VitalFile(caseid, tracks)`; the script also works with plain REST calls.
Citation: Lee H-C. et al., Sci Data 2022.

## 2. MIMIC-III Waveform Database Matched Subset (credentialed; ~10,282 patients)

```bash
export PHYSIONET_USERNAME=your_user
export PHYSIONET_PASSWORD='your_password'
python scripts/download_data.py --dataset mimic3wdb-matched --out data/mimic3wdb-matched/1.0 --sample
# full mirror (multi-TB; use the RECORDS-waveforms list to select records with ABP):
wget -r -N -c -np -nH --cut-dirs=2 --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" \
  -P data/mimic3wdb-matched https://physionet.org/files/mimic3wdb-matched/1.0/p00/
```

Records list ABP among the signals in the `.hea` header; `wfdb.rdrecord` reads them.
The matched subset links to MIMIC-III clinical data by `subject_id` and record time.

## 3. MIMIC-III clinical (credentialed): echo reports and PA-catheter cardiac output

```bash
python scripts/download_data.py --dataset mimiciii --out data/mimiciii/1.4
```

`NOTEEVENTS.csv.gz` rows with `CATEGORY = 'Echo'` contain a measurement block; parse with
`pulse_contour.references.parse_echo_measurements` (LVOT diameter, LVOT VTI, EF). Confirm the
exact field spellings on your copy before trusting the regexes. `D_ITEMS.csv.gz` +
`pulse_contour.references.cardiac_output_items` list the thermodilution / continuous cardiac
output itemids for `CHARTEVENTS.csv.gz`.

## 4. MIMIC-IV Waveform v0.1.0, MIMIC-IV-ECHO, MIMIC-IV chartevents (credentialed)

```bash
python scripts/download_data.py --dataset mimic4wdb --out data/mimic4wdb/0.1.0 --sample
python scripts/download_data.py --dataset mimic-iv-echo --out data/mimic-iv-echo --sample     # study/record lists only
python scripts/download_data.py --dataset mimiciv-icu --out data/mimiciv/3.1/icu               # d_items + chartevents
```

MIMIC-IV Waveform v0.1.0 is small (200 records); its overlap with MIMIC-IV-ECHO patients must be
checked (`subject_id` join) before planning the echo-vs-waveform arm. Later waveform releases are
expected to be much larger; re-run the overlap check when they appear.

## 5. Verify

```bash
python scripts/download_data.py --verify --out data
```
