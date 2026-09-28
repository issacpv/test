# Data acquisition — sex-stratified-brain-age

Nothing under `data/` is committed. Open cohorts can be fetched with `scripts/download_data.py`; DUA cohorts require manual steps described below.

## Open cohorts (scripted)

### IXI (open, CC BY-SA 3.0)

```bash
python scripts/download_data.py --ixi            # T1 archive (~4.5 GB) + demographics spreadsheet
python scripts/download_data.py --ixi --sample   # demographics only + first 5 T1 volumes
```

Files: `IXI-T1.tar` and `IXI.xls` from https://brain-development.org/ixi-dataset/ (the script uses the published S3/Imperial mirror URLs). Age (`AGE`), sex (`SEX_ID (1=m, 2=f)`) and site (from the filename: Guys, HH, IOP) come from `IXI.xls`.

### SALD and DLBS (INDI, open after registration)

INDI datasets are distributed from an S3 bucket (`fcp-indi`). Listing is public; the script uses anonymous S3 access:

```bash
python scripts/download_data.py --indi SALD
python scripts/download_data.py --indi DLBS --sample
```

Phenotypic CSVs (`SALD_participants.csv`, `DLBS_phenotypic.csv`) accompany the images.

### OpenNeuro (optional extra lifespan T1 datasets)

```bash
pip install openneuro-py
openneuro-py download --dataset ds000221 --include "sub-*/ses-01/anat/*T1w*"   # example
```

## Registration / DUA cohorts (manual)

### Cam-CAN

Apply at https://cam-can.mrc-cbu.cam.ac.uk/dataset/ (data request form; academic use). You receive credentials for the Cam-CAN data portal; download `cc700/mri/pipeline/release004/BIDS_20190411/anat/` T1 images and the `standard_data.csv` / `participant_data.csv` with age, sex, and cognitive/cardiovascular measures. Place under `data/camcan/`.

### HCP Young Adult (S1200)

1. Register at https://db.humanconnectome.org and accept the Open Access Data Use Terms.
2. Open data include FreeSurfer outputs and age in 5-year bins (`Age` in `unrestricted_*.csv`). Exact age (`Age_in_Yrs`) and family structure require the Restricted Data application.
3. Download via the AWS S3 bucket `hcp-openaccess` with your ConnectomeDB-linked AWS credentials:

   ```bash
   export AWS_ACCESS_KEY_ID=...; export AWS_SECRET_ACCESS_KEY=...
   aws s3 cp s3://hcp-openaccess/HCP_1200/100307/T1w/100307/stats/ data/hcp_ya/100307/stats/ --recursive
   ```

   `python scripts/download_data.py --hcp-ya --subjects data/hcp_ya/subjects.txt` wraps this loop (requires `boto3`). Retest subjects are in the `HCP_Retest` prefix.

### HCP-Aging (HCP-A)

NDA Data Use Certification (https://nda.nih.gov/ccf). Use the NDA download manager or `downloadcmd` with a package containing `HCP-A` FreeSurfer stats and `ndar_subject01`/`fsdata` tables; place under `data/hcp_a/`.

### OASIS-3

NITRC DUA (https://www.nitrc.org/projects/oasis3/), then XNAT Central. See `scripts/download_data.py --oasis3` (uses `XNAT_USER`/`XNAT_PASS`) for FreeSurfer stats and the demographics/CDR tables; or use https://github.com/NrgXnat/oasis-scripts.

## FreeSurfer tables

For every cohort produce `data/<cohort>/fs_tables/aparc_thickness_lh.tsv`, `aparc_thickness_rh.tsv`, `aparc_area_*.tsv`, `aseg_volume.tsv` with:

```bash
export SUBJECTS_DIR=data/<cohort>/freesurfer
aparcstats2table --subjects $(ls $SUBJECTS_DIR) --hemi lh --meas thickness --tablefile data/<cohort>/fs_tables/aparc_thickness_lh.tsv
asegstats2table --subjects $(ls $SUBJECTS_DIR) --meas volume --tablefile data/<cohort>/fs_tables/aseg_volume.tsv
```

and a `participants.tsv` with `participant_id, age, sex, scanner, cohort, family_id (if any), session, outcome columns`.

## Sample mode

`python scripts/download_data.py --sample-synthetic` writes simulated FreeSurfer-style tables (`data/sample/`) from `sexstrat_brainage.simulation` so the whole pipeline runs without any real data.
