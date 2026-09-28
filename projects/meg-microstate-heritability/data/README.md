# Data acquisition

## 1. HCP Young Adult (S1200) MEG - open access with registration

1. Create an account at https://db.humanconnectome.org (ConnectomeDB).
2. Accept the *WU-Minn HCP Consortium Open Access Data Use Terms* (click-through in ConnectomeDB).
3. Generate AWS credentials: ConnectomeDB -> "Amazon S3 Access" -> create key pair. Put them in
   your environment (never in the repo):

   ```bash
   export HCP_AWS_ACCESS_KEY_ID=...
   export HCP_AWS_SECRET_ACCESS_KEY=...
   ```

4. List subjects that have MEG and download resting runs:

   ```bash
   pip install boto3            # or install the aws CLI
   python scripts/download_data.py --list-subjects          # writes data/hcp_meg_subjects.txt
   python scripts/download_data.py --sample                 # one subject, one resting run
   python scripts/download_data.py --subjects-file data/hcp_meg_subjects.txt --rest --anatomy
   ```

   Equivalent AWS CLI (bucket `hcp-openaccess`, prefix `HCP_1200/<subject>/`):

   ```bash
   AWS_ACCESS_KEY_ID=$HCP_AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$HCP_AWS_SECRET_ACCESS_KEY \
     aws s3 sync s3://hcp-openaccess/HCP_1200/100307/MEG/Restin/ data/hcp/100307/MEG/Restin/
   AWS_ACCESS_KEY_ID=... aws s3 sync s3://hcp-openaccess/HCP_1200/100307/unprocessed/MEG/3-Restin/ data/hcp/100307/unprocessed/MEG/3-Restin/
   AWS_ACCESS_KEY_ID=... aws s3 sync s3://hcp-openaccess/HCP_1200/100307/MEG/anatomy/ data/hcp/100307/MEG/anatomy/
   ```

   What you need:
   - `MEG/Restin/rmegpreproc/` : HCP-cleaned resting sensor data (FieldTrip .mat), 3 runs (scan ids 3, 4, 5).
   - `unprocessed/MEG/{3,4,5}-Restin/4D/` : raw 4D files (`c,rfDC`, `config`, `hs_file`) which contain the
     ECG/EOG reference channels and the head-shape digitisation used for head-position covariates.
   - `MEG/anatomy/` : head model, source model, transformation matrices for source-space microstates.
   Read raw 4D files with `mne.io.read_raw_bti` (or the `mne-hcp` helper package).

5. Behavioural / demographic open table (age range, sex): download `unrestricted_*.csv` from ConnectomeDB
   ("Subjects" -> export) into `data/hcp/`.

## 2. HCP Restricted Data (zygosity, family structure) - separate approval

Twin modelling needs `ZygosityGT` (genotype-confirmed), `ZygositySR`, `Family_ID`, `Mother_ID`,
`Father_ID` and exact `Age_in_Yrs`. Apply at
https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-restricted-data-use-terms ,
then export `RESTRICTED_*.csv` from ConnectomeDB into `data/restricted/`. This file must never be
committed, shared, or uploaded anywhere. The heritability code reads it from
`HCP_RESTRICTED_CSV` (environment variable) so its path never appears in code.

## 3. Secondary datasets

- **Cam-CAN MEG** (resting, with ECG; ~650 unrelated adults): apply at
  https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/ ; data are delivered per agreement.
- **OMEGA** (Open MEG Archive, CTF systems): register at https://www.mcgill.ca/bic/resources/omega .
- **HCP resting fMRI** (secondary CAP-heritability comparison): same S3 bucket,
  `HCP_1200/<subject>/MNINonLinear/Results/rfMRI_REST1_LR/` etc.

## Expected layout

```
data/
  README.md
  hcp_meg_subjects.txt
  hcp/
    unrestricted_<user>_<date>.csv
    100307/
      MEG/Restin/rmegpreproc/100307_MEG_3-Restin_rmegpreproc.mat
      MEG/anatomy/...
      unprocessed/MEG/3-Restin/4D/{c,rfDC,config,hs_file}
    ...
  restricted/                     (RESTRICTED_*.csv; git-ignored; path via HCP_RESTRICTED_CSV)
  derived/
    microstates/<subject>_run<k>.npz   (labels, templates, parameters)
    hrv/<subject>_run<k>.json
    headpos/<subject>_run<k>.json
```
