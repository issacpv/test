# Data acquisition

Nothing under `data/` is committed except this file. Everything the audit reads is either a
PhysioNet **credentialed** resource (must stay on approved, encrypted storage) or an open
de-identification gold-standard corpus used only to calibrate detector recall.

Expected layout:

```
data/
  mimic-iv-note/2.2/
    discharge.csv.gz          # 331,794 discharge summaries (note_id, subject_id, hadm_id, note_type, charttime, storetime, text)
    discharge_detail.csv.gz
    radiology.csv.gz          # 2,321,355 radiology reports
    radiology_detail.csv.gz
  mimiciv/3.1/hosp/
    patients.csv.gz  admissions.csv.gz  services.csv.gz   # era (anchor_year_group), service, demographics for stratified rates
  mimiciii/1.4/
    NOTEEVENTS.csv.gz         # optional: MIMIC-III notes de-identified with the rule-based `deid` pipeline (method comparison)
  deid-gold/                  # PhysioNet `deid` package (open): gold-standard nursing-note corpus with surrogate PHI
  n2c2-2014/                  # optional: i2b2/n2c2 2014 de-identification corpus (DUA, free registration)
```

## 1. Open calibration corpus: PhysioNet `deid` gold standard

The `deid` software package (Neamatullah et al., BMC Med Inform Decis Mak 2008) ships with a
gold-standard corpus of nursing notes in which PHI was replaced by realistic *surrogates* and
annotated. It is open (no credentialing):

```bash
python scripts/download_data.py --dataset deid-gold --out data/deid-gold
```

The script fetches the package listing from `https://physionet.org/files/deid/1.1/` and downloads
the corpus/annotation files. Use it to measure detector recall on realistic (surrogate) PHI in
addition to the planted-identifier calibration in `deid_audit.estimate`.

## 2. MIMIC-IV-Note v2.2 (PhysioNet credentialed)

1. PhysioNet account + CITI "Data or Specimens Only Research" + signed DUA for **MIMIC-IV-Note**
   (https://physionet.org/content/mimic-iv-note/2.2/) and **MIMIC-IV** (for era/service covariates).
2. Export credentials (never commit them):

```bash
export PHYSIONET_USERNAME=your_user
export PHYSIONET_PASSWORD='your_password'
```

3. Download:

```bash
python scripts/download_data.py --dataset mimic-iv-note --out data/mimic-iv-note/2.2
python scripts/download_data.py --dataset mimiciv-hosp --out data/mimiciv/3.1/hosp
# equivalent wget:
wget -N --user "$PHYSIONET_USERNAME" --password "$PHYSIONET_PASSWORD" -P data/mimic-iv-note/2.2 \
  https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz
```

Text columns: `discharge.text`, `radiology.text`; PHI placeholders are `___`; dates are shifted
into 2100-2200 consistently per patient.

## 3. MIMIC-III v1.4 notes (optional, credentialed)

```bash
python scripts/download_data.py --dataset mimiciii-notes --out data/mimiciii/1.4
```

`NOTEEVENTS.csv.gz` (`CATEGORY`, `TEXT`); placeholders are tagged, e.g. `[**First Name (STitle) 123**]`,
`[**2150-3-12**]`, `[**Hospital1 18**]`, which `deid_audit.sections.mimic3_placeholder_types` histograms.

## 4. i2b2/n2c2 2014 de-identification corpus (optional)

Register at the n2c2 NLP data portal (https://portal.dbmi.hms.harvard.edu/) and sign the DUA for
"2014 De-identification and Heart Disease Risk Factors Challenge". Place the XML files under
`data/n2c2-2014/`. This is a second realistic-surrogate gold standard for recall calibration.

## 5. Running the audit

```bash
PYTHONPATH=src python -m deid_audit.audit --input data/mimic-iv-note/2.2/discharge.csv.gz \
    --note-type discharge --ner-model en_core_web_sm --out outputs/discharge_counts.csv
```

writes per-note *count* rows and a summary table of rates per 10k notes; no text is written.
All processing is local; do not point any step at a hosted API.
