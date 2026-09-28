# Data acquisition

Nothing under `data/` is committed except this file.

## 0. Credentials (clinical databases only)

MIMIC-III v1.4 and MIMIC-IV v3.1 clinical data need PhysioNet credentialing (account, CITI "Data or Specimens
Only Research", DUA per project page). The waveform databases below are open, but their records link to
credentialed patients, so treat everything you derive as credentialed.

```bash
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"
```

## 1. Smoke test (no credentials)

```bash
pip install wfdb            # needed for waveform downloads
python scripts/download_data.py --sample
```

This fetches: the MIMIC-IV demo (clinical schema for `inputevents`), the first few MIMIC-III matched
waveform records (segments listed in each record header), and 2 VitalDB cases that have both an arterial
waveform (`SNUADC/ART`) and a device SVV track (`EV1000/SVV` or `Vigileo/SVV`).

## 2. Full downloads

```bash
python scripts/download_data.py --mimic3-clinical        # INPUTEVENTS_MV, CHARTEVENTS, D_ITEMS, ICUSTAYS, ... (credentialed)
python scripts/download_data.py --mimic4-clinical        # inputevents, chartevents, d_items, icustays (credentialed)
python scripts/download_data.py --mimic4-waveforms       # MIMIC-IV Waveform DB v0.1.0 records (open)
python scripts/download_data.py --vitaldb --max-cases 200
```

MIMIC-III matched waveforms are **not** downloaded wholesale (tens of TB). The analysis code fetches only the
+/- 60 min around each bolus with `wfdb.rdrecord(..., pn_dir='mimic3wdb-matched/1.0/pXX/pXXXXXX', sampfrom=,
sampto=)` and caches it under `data/waveform_cache/`.

Manual equivalents:

```bash
# MIMIC-III clinical tables (credentialed)
for t in INPUTEVENTS_MV CHARTEVENTS D_ITEMS ICUSTAYS PATIENTS ADMISSIONS; do
  wget -N -c --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
       https://physionet.org/files/mimiciii/1.4/$t.csv.gz -P data/raw/mimiciii/1.4/
done
# MIMIC-III matched waveform record list (open)
wget -N https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms -P data/raw/mimic3wdb-matched/1.0/
# MIMIC-IV waveform record list (open)
wget -N https://physionet.org/files/mimic4wdb/0.1.0/RECORDS -P data/raw/mimic4wdb/0.1.0/
```

VitalDB open API (no key): `https://api.vitaldb.net/cases` (clinical info), `https://api.vitaldb.net/trks`
(track list: `tid, caseid, tname`), `https://api.vitaldb.net/<tid>` (track values as CSV). The `vitaldb`
package wraps these (`vitaldb.load_case(caseid, ['SNUADC/ART', 'EV1000/SVV'])`).

## 3. Expected layout

```
data/
  README.md
  raw/
    mimic-iv-demo/2.2/{hosp,icu}/*.csv.gz
    mimiciii/1.4/*.csv.gz
    mimiciv/3.1/icu/*.csv.gz
    mimic3wdb-matched/1.0/RECORDS-waveforms, pXX/pXXXXXX/*.hea|*.dat   (sample records only)
    mimic4wdb/0.1.0/...
    vitaldb/cases.csv, trks.csv, tracks/<tid>.csv
  waveform_cache/<source>/<subject>/<bolus_id>.npz
  derived/
    boluses.parquet      # (stay_id, subject_id, bolus_id, start, end, amount_ml, item, isolated, vaso_change)
    responses.parquet    # pre/post MAP, PP, HR stats, labels, sham flag, prerequisite flags
    ppv.parquet          # per bolus: ppv_median, spv_median, sqi_fraction, rr_irregularity, rr_from_pp
```

## 4. Item ids

Fluid, vasopressor, ventilator-setting and cardiac-output item ids are resolved by regex over `D_ITEMS.LABEL`
/ `d_items.label` at run time (`fluid_resp.boluses.resolve_itemids`) and saved with each run.
