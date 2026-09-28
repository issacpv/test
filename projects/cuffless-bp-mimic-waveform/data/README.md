# Data acquisition

Nothing under `data/` is committed. The waveform databases are large; the loader in
`src/cuffless_bp/wfdb_loader.py` streams records directly from PhysioNet so a full local copy is
not required. Cache only the 10-s windows you keep (`data/windows/*.npz`).

## 1. MIMIC-III Waveform Database Matched Subset v1.0 (open access)

No credentialing is needed for the waveforms. Record names look like
`p00/p000020/p000020-2183-04-28-17-47` (patient-level multi-segment records) and the numerics
records end in `n`. `subject_id` is the number after `p` (MIMIC-III ids).

```bash
# list records (RECORDS file, ~22k lines)
curl -sS https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS -o data/mimic3wdb_matched_RECORDS
# stream one record from Python
python - <<'EOF'
import wfdb
hdr = wfdb.rdheader('p000020-2183-04-28-17-47', pn_dir='mimic3wdb-matched/1.0/p00/p000020')
print(hdr.seg_name[:5], hdr.fs)
EOF
```

`scripts/download_data.py --db mimic3wdb --sample` fetches the RECORDS index and the headers of the
first few records with ECG + PLETH + ABP and writes `data/mimic3wdb/index.csv`.

## 2. MIMIC-IV Waveform Database v0.1.0 (open access) + MIMIC-IV v3.1 clinical (credentialed)

Waveforms: https://physionet.org/content/mimic4wdb/0.1.0/ (records under `waves/pXXX/pXXXXXXXX/<record>/`;
`subject_id` is the 8-digit number and matches MIMIC-IV). The record header `base_datetime` is on the
same shifted time axis as MIMIC-IV clinical timestamps, which is what makes bolus linkage possible.

```bash
python scripts/download_data.py --db mimic4wdb --sample     # RECORDS index + first headers
```

Bolus events need `icu/inputevents.csv.gz` and `icu/icustays.csv.gz` from MIMIC-IV v3.1 (credentialed):

1. PhysioNet account + CITI "Data or Specimens Only Research" + sign the MIMIC-IV DUA.
2. `export PHYSIONET_USER=... PHYSIONET_PASS=...`
3. `python scripts/download_data.py --db mimiciv-inputevents`

which runs
`wget -N -c --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/mimiciv/3.1/icu/inputevents.csv.gz -P data/mimiciv/`.

## 3. VitalDB (open, CC BY 4.0)

```bash
pip install vitaldb
python scripts/download_data.py --db vitaldb --sample   # case list + 3 cases, 60 s each
```

The `vitaldb` package reads the open API (https://api.vitaldb.net). Track names used here:
`SNUADC/ECG_II`, `SNUADC/PLETH`, `SNUADC/ART` (500 Hz waveforms); case metadata from
`https://api.vitaldb.net/cases`; drug/pump tracks are listed per case in `https://api.vitaldb.net/trks`
(e.g. `Orchestra/PHEN_RATE`, `Orchestra/NEPI_RATE`; check the track list for the exact names in each case).
Accept the dataset terms on https://vitaldb.net/dataset/ before large downloads.

## 4. PulseDB (open)

Follow https://github.com/pulselabteam/PulseDB (links to the `.mat` v7.3 files, ~100 GB total). Put the
files under `data/pulsedb/` and read them with `cuffless_bp.wfdb_loader.load_pulsedb_file` (h5py).
PulseDB's `CalibFree` and `AAMI` test-set definitions are used only as an external cross-check of our own
subject-level splits.

## Expected layout

```
data/
  mimic3wdb/index.csv              # record, subject_id, has_ecg, has_pleth, has_abp, fs, n_samples
  mimic4wdb/index.csv
  mimiciv/inputevents.csv.gz       # credentialed
  mimiciv/icustays.csv.gz          # credentialed
  vitaldb/cases.csv                # open
  vitaldb/case_<id>.npz            # cached waveforms (optional)
  pulsedb/*.mat                    # optional
  windows/<source>/<subject>/<record>_<start>.npz   # accepted 10-s windows + labels
```
