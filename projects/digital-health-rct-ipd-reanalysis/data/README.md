# Data acquisition

Nothing in this folder is committed except this file. There are two layers: the **registry** (open, automatic) and the **individual participant data** (by application or from open repositories; never leaves the approved analysis environment).

## Expected layout

```
data/
  registry/
    studies.csv                # ClinicalTrials.gov v2 API extract: one row per trial with IPD-sharing fields
    sharing_summary.csv        # share of trials with IPD sharing = YES by sponsor class (Wilson CIs)
    screening.csv              # manual: nct_id, digital_confirmed, intervention_class (app / DTx / remote-monitoring / iCBT / SMS), requested_from, request_date, outcome
  ipd/                         # never committed; usually lives only inside the provider's secure environment
    <nct_id>/raw/              # as delivered (SAS/CSV/XPT)
    <nct_id>/tidy.parquet      # harmonised: trial, id, z, baseline, age, sex, engaged, dose, y (+ engagement telemetry table)
    <nct_id>/engagement_events.parquet   # timestamped app-usage events if provided
  open/                        # IPD from open repositories (OSF / Zenodo / Dryad / ICPSR), with licence files
```

## 1. Registry (ClinicalTrials.gov v2 API, open)

```bash
python scripts/download_data.py --sample
python scripts/download_data.py --start-year 2015
python scripts/download_data.py --ipd-yes-only
```

The v2 API (https://clinicaltrials.gov/data-api/api) is queried with `query.intr=<term>` for each digital-health term (`dh_ipd.registry.DIGITAL_TERMS`), `filter.overallStatus=COMPLETED`, and the fields of the `ipdSharingStatementModule` (`ipdSharing`, `description`, `infoTypes`, `timeFrame`, `accessCriteria`, `url`). Pagination uses `nextPageToken`; no key is needed. `studies_to_frame` de-duplicates by NCT id, flags digital interventions by keyword (`is_digital_intervention`) and classifies the repository named in the statement (`classify_repository`: vivli / yoda / csdr / nda / osf / zenodo / dryad / icpsr / github / on_request / none). Screen the digital flag manually for the analysis set (`screening.csv`).

## 2. Individual participant data (by application)

| Platform | What is there | How to apply | Environment |
|---|---|---|---|
| Vivli (https://vivli.org) | Trials contributed by pharma, device and academic members; search by NCT id, sponsor, intervention | Research proposal + SAP + team CVs; review by the data contributor; data-use agreement | Secure research environment (no download; analysis code is uploaded; outputs vetted) |
| YODA Project (https://yoda.yale.edu) | Johnson & Johnson and other partner trials, incl. medical-device trials | Proposal via the YODA form; independent review; DUA | Secure environment |
| ClinicalStudyDataRequest.com (CSDR) | Remaining sponsors not migrated to Vivli | Proposal + DUA | Secure environment |
| NIMH Data Archive (NDA; https://nda.nih.gov) | NIMH-funded trials, including many digital mental-health RCTs, with participant-level data | Institutional NDA data-access request (signing official), per collection | Download to approved institutional storage |
| Open repositories (OSF, Zenodo, Dryad, ICPSR) | IPD shared alongside papers in JMIR, Lancet Digital Health, npj Digital Medicine, etc. | None beyond the licence | Local |

Workflow: (1) from `studies.csv`, list digital trials with `ipd_yes` and a named repository; (2) confirm the trial is actually listed on the platform (Vivli/YODA search) and record it in `screening.csv`; (3) submit one proposal per platform covering all eligible trials, with the pre-registered SAP (`docs/` in the paper repo); (4) harmonise each delivered dataset to the tidy schema below; (5) analyses run inside the secure environment with this package installed from source (`pip install -e .` is usually permitted); only aggregate outputs leave.

## 3. Tidy IPD schema (one row per randomised participant)

| column | meaning |
|---|---|
| `trial` | NCT id |
| `id` | participant id (trial-specific, pseudonymous) |
| `z` | 1 = digital intervention arm, 0 = control (specify control type in `screening.csv`: waitlist / TAU / attention-control app) |
| `baseline` | primary outcome at baseline, standardised within trial |
| `age`, `sex` | covariates present in essentially every trial |
| `engaged` | pre-specified binary engagement (e.g. >= 1 module completed; trial-specific definition recorded) |
| `dose` | continuous engagement (sessions, modules, active days) |
| `y` | primary outcome at the primary endpoint, standardised (higher = better), NaN if missing |
| `dropout` | 1 if `y` missing |
| optional `y_t1..y_tk` | intermediate assessments for early-response analyses |

Engagement telemetry (timestamped events) is kept in a separate long table and summarised to `dose` and to first-two-weeks features for H5.

## 4. Simulated data for development

`dh_ipd.simulate.simulate_ipd()` produces a multi-trial dataset with engagement confounding, principal strata and MNAR dropout so that the entire analysis pipeline (registry excluded) runs and is unit-tested without any real IPD.

## Governance

Each platform's DUA governs storage, publication and re-identification restrictions; no IPD or participant-level derived tables are ever committed or moved outside the approved environment; only aggregate results (estimates, CIs, counts >= 10) are exported. IPD must not be sent to third-party APIs.
