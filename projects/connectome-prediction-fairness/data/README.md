# Data acquisition — `connectome-prediction-fairness`

Nothing under `data/` is committed. Expected final layout:

```
data/
├── hcp/
│   ├── unrestricted_<user>_<date>.csv       # ConnectomeDB behavioural export
│   ├── restricted_<user>_<date>.csv         # requires Restricted Data application
│   ├── subjects.txt                         # one subject ID per line
│   └── <subject>/rfMRI_REST{1,2}_{LR,RL}/
│       ├── rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
│       ├── rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.Schaefer400.ptseries.nii  (you create)
│       └── Movement_RelativeRMS_mean.txt
├── parcellations/
│   └── Schaefer2018_400Parcels_7Networks_order.dlabel.nii
├── ds002785/                                # AOMIC-PIOP1 (OpenNeuro)
│   ├── participants.tsv
│   └── derivatives/fmriprep/sub-XXXX/func/*task-restingstate*desc-confounds_*.tsv
├── ds002790/                                # AOMIC-PIOP2 (optional)
└── abcd/                                    # optional, NDA-controlled
```

## 1. HCP S1200 (free registration; imaging via AWS S3)

1. Register at <https://db.humanconnectome.org> and accept the **WU-Minn HCP Open Access Data Use Terms**.
2. In ConnectomeDB open the *WU-Minn HCP Data – 1200 Subjects* project, click **Behavioral Data → Download CSV**
   (this is the *unrestricted* table; it contains `Subject`, `Gender`, `Age` (5-yr bins), NIH Toolbox scores such
   as `CogTotalComp_Unadj`, `CogFluidComp_Unadj`). Save to `data/hcp/`.
3. Apply for **Restricted Data** (<https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage>):
   after approval a second CSV becomes available with `Family_ID`, `Mother_ID`, `Father_ID`, `ZygosityGT`,
   `Race`, `Ethnicity`, `SSAGA_Income`, `SSAGA_Educ`, `Age_in_Yrs`. Family structure is **required** for
   leakage-free CV; race/ethnicity/SES are required for the audit.
4. Create AWS keys: ConnectomeDB profile → *Amazon S3 access* → *Create credentials*. Export them:
   ```bash
   export HCP_AWS_ACCESS_KEY_ID=...
   export HCP_AWS_SECRET_ACCESS_KEY=...
   ```
5. Download rest runs (ICA-FIX cleaned, MSMAll-aligned dense time series, ~1 GB/run):
   ```bash
   python scripts/download_data.py --hcp --subjects data/hcp/subjects.txt
   # equivalently with the AWS CLI:
   aws s3 cp s3://hcp-openaccess/HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii data/hcp/100307/rfMRI_REST1_LR/
   ```
6. Parcellate with Connectome Workbench (Schaefer-400 or Glasser-360):
   ```bash
   wb_command -cifti-parcellate <dtseries> data/parcellations/Schaefer2018_400Parcels_7Networks_order.dlabel.nii COLUMN <out.ptseries.nii>
   ```
   Alternative with much smaller downloads: the **HCP1200 Parcellation+Timeseries+Netmats (PTN)** release on
   ConnectomeDB provides ICA node time series (d = 15…300) for ~1,000 subjects as text files; `cpm_fair.fc_loader.load_text_timeseries`
   reads them directly.
7. Task fMRI (optional, for the "which state predicts more fairly" analysis): same S3 layout under
   `MNINonLinear/Results/tfMRI_<TASK>_{LR,RL}/`.

## 2. AOMIC-PIOP1 / PIOP2 (OpenNeuro, fully open)

AOMIC ships fMRIPrep derivatives (v1.3.2) inside the dataset. The public S3 bucket needs no credentials:

```bash
python scripts/download_data.py --aomic ds002785 --n-subjects 216 --include-bold
# or
pip install openneuro-py && openneuro-py download --dataset ds002785 --include "derivatives/fmriprep/*task-restingstate*" --include participants.tsv
# or
aws s3 sync --no-sign-request s3://openneuro.org/ds002785/derivatives/fmriprep data/ds002785/derivatives/fmriprep --exclude "*" --include "*task-restingstate*"
```

`participants.tsv` columns (verified): `participant_id, age, sex, BMI, handedness, education_category, religious_now, raven_score, NEO_*`.
`raven_score` is the cognition target, `sex` and `education_category` (academic / applied) the audited attributes.
AOMIC has **no race/ethnicity variable** (Dutch university sample), so it replicates the sex/SES arm only.

## 3. Cross-dataset replication with race/ethnicity (DUA)

* **Healthy Brain Network** (Child Mind Institute): rest fMRI + WISC-V, race/ethnicity and household income; requires a
  Data Usage Agreement (<https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/>).
* **NKI-Rockland Sample (Enhanced)**: lifespan rest fMRI + cognitive battery, race/ethnicity; DUA via NITRC.
* **ABCD** (NDA): the largest option; requires an NDA Data Use Certification. Fetch with `nda-tools`
  (`downloadcmd -dp <package_id>`) after creating a data package on the NDA site. Set `NDA_USERNAME` / `NDA_PASSWORD`.

## 4. Sanity checks after download

```bash
python -c "from cpm_fair.fc_loader import load_hcp_subject_table as f; df=f('data/hcp/unrestricted.csv','data/hcp/restricted.csv'); print(df['Race'].value_counts())"
```

Expected HCP S1200 race distribution (restricted table, n≈1,200): ~75% White, ~13% Black or African American,
~6% Asian/Nat. Hawaiian/Other Pacific Is., remaining Unknown/More than one; ~9% Hispanic/Latino ethnicity.
Verify exact counts from your own export — they change slightly with data-release updates.
