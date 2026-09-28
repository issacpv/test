# Data acquisition — cvr-normative-maps

All data live under `data/` (git-ignored).

## 1. EuskalIBUR — OpenNeuro ds003192 (open, CC0)

Multi-echo breath-hold fMRI with end-tidal CO2, 10 weekly sessions, 7 public participants.

```bash
python scripts/download_data.py --source openneuro --dataset ds003192 --sample        # 1 subject, 1 session: breath-hold echoes + physio + T1w
python scripts/download_data.py --source openneuro --dataset ds003192                 # everything (~60 GB)
# alternatives: pip install openneuro-py && openneuro-py download --dataset ds003192
#               aws s3 sync --no-sign-request s3://openneuro.org/ds003192 data/openneuro/ds003192
```

Expected layout (BIDS):

```
data/openneuro/ds003192/
  sub-001/ses-01/func/sub-001_ses-01_task-breathhold_echo-1_bold.nii.gz  (echo-1..5)
  sub-001/ses-01/func/sub-001_ses-01_task-breathhold_physio.tsv.gz + .json   (CO2, O2, respiration, cardiac)
  sub-001/ses-01/func/sub-001_ses-01_task-rest_run-*_bold.nii.gz
  sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
  task-breathhold_events.tsv / *_bold.json (timing of holds)
```

## 2. NKI-Rockland Sample enhanced (imaging public on S3; phenotypes under DUA)

1. Imaging (no credentials):

```bash
aws s3 ls --no-sign-request s3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest/
python scripts/download_data.py --source nki --sample                    # 1 subject: breath-hold + rest + T1w
python scripts/download_data.py --source nki --subjects sub-A00008326 sub-A00008399 --tasks BREATHHOLD rest
```

The breath-hold runs are named like `sub-*/ses-*/func/sub-*_ses-*_task-BREATHHOLD_acq-1400_bold.nii.gz`; resting runs `task-rest_acq-645` and `task-rest_acq-1400`. Physiological logs, when present, sit next to the run as `*_physio.tsv.gz`.

2. Phenotypes (age, sex, vitals, medical history, medications) and the full physiological logs require the NKI-RS Data Use Agreement (http://fcon_1000.projects.nitrc.org/indi/enhanced/ → "Data access"). After approval you receive the assessment CSVs and download links; put them under `data/nki/phenotypes/` (git-ignored). Set `NKI_PHENO_DIR` if you keep them elsewhere.

Expected layout:

```
data/nki/
  sub-A00008326/ses-BAS1/func/sub-A00008326_ses-BAS1_task-BREATHHOLD_acq-1400_bold.nii.gz
  sub-A00008326/ses-BAS1/func/sub-A00008326_ses-BAS1_task-rest_acq-645_bold.nii.gz
  sub-A00008326/ses-BAS1/anat/sub-A00008326_ses-BAS1_T1w.nii.gz
  phenotypes/*.csv   (DUA)
```

## 3. Other OpenNeuro breath-hold / CO2 datasets (optional)

Use OpenNeuro's search ("breath-hold", "hypercapnia", "CO2", "cerebrovascular reactivity") and download with the same script (`--dataset dsXXXXXX`). Check each dataset's physio JSON for the presence of a CO2 channel.

## 4. Atlases

- Schaefer-200 (nilearn `fetch_atlas_schaefer_2018`), arterial territory atlas (Liu et al., 2023, Scientific Data; NITRC), MNI GM/WM tissue priors (fMRIPrep templates).

## Derived products

```
outputs/
  cvr/<dataset>/<sub>_<ses>_cvr-amplitude.nii.gz, *_cvr-delay.nii.gz, *_r2.nii.gz, *_parcels.tsv
  physio/<dataset>/<sub>_<ses>_petco2.tsv, *_rvt.tsv, *_compliance.json
  normative/centiles_<region>.csv, params.json
  reliability/euskalibur_icc.csv
```
