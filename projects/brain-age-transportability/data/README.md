# Data acquisition: brain-age-transportability

All three cohorts require registration and acceptance of data-use terms.
**Nothing under `data/` is committed** (see `.gitignore`). Credentials are read
from environment variables by `scripts/download_data.py`; never hard-code them.

Expected layout after download:

```
data/
  hcp_ya/
    subjects.csv                       # from ConnectomeDB export (see 1.)
    restricted.csv                     # ONLY if you hold Restricted Access (exact age)
    stats/<subject>/{aseg,lh.aparc,rh.aparc}.stats
  hcp_lifespan/
    hcp_aging/   ndar_subject01.txt + FreeSurfer stats per session
    hcp_dev/     ndar_subject01.txt + FreeSurfer stats per session
  oasis3/
    sessions.csv                       # MR session list incl. scanner (XNAT export)
    adrc_clinical.csv                  # ADRC Clinical Data (CDR, MMSE, dx, ageAtEntry)
    freesurfer/<OASxxxxx_Freesurfer53_dXXXX>/DATA/<session>/stats/*.stats
```

Only the FreeSurfer `stats/` tables (~100 kB per session) are needed for the
starter pipeline; raw NIfTI/DICOM is optional (only for re-running FreeSurfer
with one version across cohorts, see README "Risks").

---

## 1. HCP Young Adult (S1200) — free registration + Open Access Data Use Terms

1. Create an account at https://db.humanconnectome.org (ConnectomeDB).
2. Accept the **WU-Minn HCP Open Access Data Use Terms** (click-through).
3. Age caveat: the *open* data give age only in bins (`22-25`, `26-30`, `31-35`,
   `36+`). Exact age (`Age_in_Yrs`) is in the **Restricted Data** tier, which
   needs a separate application (PI signature). Exact age is strongly
   preferred for training a brain-age model; with bins only, use bin midpoints
   and treat HCP-YA as a *test* cohort rather than a training cohort.
4. Export the subject table: ConnectomeDB → `WU-Minn HCP Data - 1200 Subjects`
   → "Open Access" → Download CSV → save as `data/hcp_ya/subjects.csv`. This
   CSV already contains FreeSurfer summary columns (`FS_*_Vol`, `FS_*_Thck`)
   which are sufficient for a first model.
5. Per-subject FreeSurfer `stats/` files are on AWS S3. In ConnectomeDB, under
   "Amazon S3 Access", generate AWS credentials, then:

   ```bash
   export HCP_AWS_ACCESS_KEY_ID=...
   export HCP_AWS_SECRET_ACCESS_KEY=...
   python scripts/download_data.py hcp-ya --sample 10      # 10 subjects
   python scripts/download_data.py hcp-ya                  # all in subjects.csv
   ```

   which fetches `s3://hcp-openaccess/HCP_1200/<subject>/T1w/<subject>/stats/{aseg,lh.aparc,rh.aparc}.stats`
   (FreeSurfer 5.3-HCP). Equivalent AWS CLI:
   `aws s3 cp s3://hcp-openaccess/HCP_1200/100307/T1w/100307/stats/aseg.stats .`

## 2. HCP-Aging and HCP-Development (Lifespan) — NDA Data Use Certification

1. Create an NDA account at https://nda.nih.gov and have your institution's
   signing official co-sign a **Data Use Certification** for the
   "Connectomes Related to Human Disease / Lifespan HCP" permission group.
2. On NDA, add the collections to a *data package*:
   **HCP-Aging (collection 2847)** and **HCP-Development (collection 2846)**.
   Include `ndar_subject01` (has `interview_age` in months, `sex`) and the
   structural-preprocessed / FreeSurfer outputs (`stats/` tables) for the
   Lifespan 2.0 release.
3. Download with `nda-tools`:

   ```bash
   pip install nda-tools
   export NDA_USERNAME=... ; export NDA_PASSWORD=...
   downloadcmd -dp <PACKAGE_ID> -d data/hcp_lifespan -u $NDA_USERNAME -p $NDA_PASSWORD
   ```

   `scripts/download_data.py hcp-lifespan` prints these steps and validates the
   resulting layout; it does not bypass NDA.
4. Scanner: Siemens Prisma 3T at four sites, 0.8 mm MPRAGE (Harms et al., 2018).

## 3. OASIS-3 — free registration + Data Use Agreement (XNAT Central)

1. Apply at https://www.oasis-brains.org (accept the DUA); access is granted to
   the `OASIS3` project on https://central.xnat.org.
2. From the XNAT project page export CSVs (Manage Files / spreadsheet download):
   - **MR Sessions** (`OASIS3_MR_sessions.csv`): `label` (e.g. `OAS30001_MR_d0129`),
     `Scanner` (Siemens TIM Trio 3T, Biograph mMR 3T, Vision/Sonata 1.5T), `Age`.
   - **ADRC Clinical Data** (`OASIS3_ADRC_clinicaldata.csv`): `ADRC_ADRCCLINICALDATA ID`
     (`OAS30001_ClinicalData_d0000`), `ageAtEntry`, `cdr`, `mmse`, `dx1`, `apoe`.
   - Optionally **UDS B4/C1** forms for CDR sum-of-boxes / neuropsych.
   Save them as `data/oasis3/sessions.csv` and `data/oasis3/adrc_clinical.csv`.
   The `dXXXX` suffix is days since the participant's baseline; it is the key
   for aligning scans and clinical visits (`longitudinal.align_clinical_to_scans`).
3. FreeSurfer 5.3-HCP outputs are provided as XNAT *assessors*
   (`OAS30001_Freesurfer53_d0129`). Download the `stats/` tables with the REST API:

   ```bash
   export OASIS_USER=... ; export OASIS_PASS=...
   python scripts/download_data.py oasis3 --sample 5
   python scripts/download_data.py oasis3
   ```

   This mirrors the official `download_freesurfer.sh` from
   https://github.com/NrgXnat/oasis-scripts (endpoint
   `/data/archive/projects/OASIS3/subjects/<subj>/experiments/<fs_id>/files?format=zip`).
   Raw T1w scans, if ever needed: `download_oasis_scans.sh` from the same repo.

## 4. Ethics reminders

- OASIS-3 DUA and HCP terms forbid re-identification and redistribution;
  keep participant-level tables outside version control.
- HCP Restricted data (exact age, family structure) must not be merged into
  outputs that are shared publicly at individual level.
- NDA data may not be shared beyond the certified users named on the DUC.
