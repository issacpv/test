# radiology-delay-ed-disposition

**Reports that arrive after the patient has left: radiology report turnaround (storetime minus charttime) in MIMIC-IV-Note linked to MIMIC-IV-ED stays — how often ED disposition precedes the final report, whether turnaround delays post-imaging ED dwell time and admission, and whether post-exit finalisation and addenda predict 72-hour returns, using time-of-day / weekend variation as quasi-experimental leverage.**

## Status / difficulty / timeline / compute

- Status: design + starter code (report-to-ED-stay linkage, modality classification, turnaround and post-exit flags with addendum handling, timing features and CXR severity, adjusted models, manual 2SLS with robust SEs, regression discontinuity in time-of-day, negative-control checks). No data is shipped.
- Difficulty: MSc-level (clinical informatics / health-services research); 5-7 months.
- Compute: workstation; MIMIC-IV-Note `radiology.csv.gz` is ~2.3M reports (a few GB uncompressed); pandas/DuckDB, no GPU.

## Background

Radiology turnaround time (TAT) is a standard operational metric (Boland et al., 2008 Eur Radiol) and is repeatedly linked to ED crowding and length of stay in single-centre operational studies (EMRA literature review; an AJR study of acute-abdomen CT workflow, doi:10.2214/AJR.14.14057, found CT-related intervals accounted for ~29% of ED LOS with radiology TAT ~32% of that interval). Overnight reading is associated with fatigue-related performance changes (Hanna et al., 2018 JACR), and EDs act on preliminary or clinician reads before the final report. Yet the patient-level consequences of report delay — longer post-imaging dwell, disposition made without the final read, discharge with a pending report, addenda after discharge — have not been quantified on open data.

MIMIC-IV-Note v2.2 (Johnson et al., PhysioNet 2023) contains ~2.3M radiology reports for MIMIC-IV patients, each with `charttime` (exam/charting time) and `storetime` (time the report was signed and stored), a `note_type` distinguishing reports (RR) from addenda (AR), and a `radiology_detail` table carrying exam name, exam code, CPT code and addendum-parent links. MIMIC-IV-ED v2.2 gives ED arrival, departure and disposition for ~425k ED stays of the same patients, and MIMIC-CXR (Johnson et al., 2019 Sci Data) adds CheXpert labels for chest radiographs. A 2026 benchmark paper (EarlyDx, arXiv:2607.28788) noted in passing that using `storetime` instead of `charttime` as the availability time removes 19.3% of radiology reports from an admission-anchored window — a direct measurement of how often reports finalise after the clinical decision point, which has never been studied as a phenomenon in itself.

## The research gap

What has been done (2023-2026):

- Operational TAT studies are single-centre, use aggregate TAT, and do not link to patient-level outcomes or to the timing of the disposition decision.
- MIMIC-IV-ED benchmark and ED-LLM work (Xie et al., 2022 Sci Data; ED-Copilot, arXiv:2402.13448) treat imaging as a feature or a cost, not as a timed process; MIMIC-CXR report-generation work ignores report timing.
- A 2024 JACR paper (S1546-1440(24)00939-6) describes deploying an ED radiology workflow tool; again operational, closed data.
- The EarlyDx observation (19.3% of reports finalise after the anchor) is the only quantitative hint on MIMIC; it was not analysed by modality, disposition or outcome.

What is specifically missing (our angle):

1. **Decision-before-report prevalence**: the share of ED stays with imaging whose disposition (ED `outtime`) precedes the report `storetime`, by modality (CXR, CT head, CT abdomen/pelvis, CTA chest, extremity radiograph, ultrasound, MRI), hour of day, weekend and disposition.
2. **Patient-level consequences of TAT**: post-imaging ED dwell (outtime minus charttime), admission, ICU within 24 h and 72-h ED return as functions of TAT, adjusted for acuity, age, sex, arrival mode, chief-complaint cluster, modality, exam count, and CXR severity (CheXpert positive-finding count) — with explicit handling of the prioritisation confound (sicker patients get faster reads).
3. **Discharged with a pending report**: among discharged patients, the prevalence of post-exit finalisation and of addenda after discharge, and their association with 72-h return and subsequent admission — the safety-relevant tail.
4. **Quasi-experimental leverage** that is feasible under MIMIC's per-patient date shifting (which destroys cross-patient queue reconstruction but preserves time-of-day and day-of-week): hour-of-day and weekend effects on TAT within modality, a regression-discontinuity-in-time design at reading-shift boundaries, and negative-control outcomes (pre-imaging wait; arrival-to-exam time) that must not depend on TAT.
5. **Preliminary reads**: text markers ("wet read", "preliminary", "discussed with", "critical result communicated") as a moderator: does a documented communication attenuate the effect of a slow final report?

Related projects in this repo: `ed-triage-mimic-ed` (triage prediction) shares the ED cohort machinery. This project is self-contained.

## Research questions / hypotheses

1. **RQ1 (prevalence).** H1: the ED disposition precedes the final report in >= 15% of ED stays with imaging overall, > 30% for chest radiographs at night, and < 10% for CT head; the fraction is higher for discharged than for admitted patients.
2. **RQ2 (dwell).** H2: each additional hour of TAT (within modality, adjusted) is associated with >= 20 min longer post-imaging ED dwell for CT abdomen/pelvis and CTA chest, but not for chest radiographs (which are acted on before the report). The 2SLS estimate using hour-of-day x modality instruments is of the same sign and within 2x of the OLS estimate.
3. **RQ3 (disposition).** H3: longer TAT increases the odds of admission for intermediate acuity (ESI 3) because clinicians admit rather than wait; no effect at ESI 1-2.
4. **RQ4 (pending report at discharge).** H4: discharged patients whose report finalised after ED exit have higher adjusted odds of 72-h ED return (OR >= 1.15); an addendum after discharge raises it further (OR >= 1.5), consistent with missed-finding recall.
5. **RQ5 (time-of-day).** H5: TAT for CT rises by >= 30% between 23:00 and 07:00 and on weekends, with a discontinuity at the day-shift start; post-imaging dwell shows the same discontinuity (the RDiT estimate has the same sign as the OLS effect).
6. **RQ6 (communication).** H6: reports with documented verbal communication ("discussed with", "critical result") show no dwell-time penalty from TAT.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV-Note v2.2 | `radiology` (charttime, storetime, note_type RR/AR, text) and `radiology_detail` (exam_name, exam_code, cpt_code, parent_note_id / addendum links) | ~2.3M reports | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimic-iv-note/2.2/ |
| MIMIC-IV-ED v2.2 | edstays (intime, outtime, disposition, arrival_transport), triage (acuity, chiefcomplaint), diagnosis | ~425k stays | PhysioNet credentialed | https://physionet.org/content/mimic-iv-ed/2.2/ |
| MIMIC-IV v3.1 (hosp, icu) | patients (age), admissions (hospital_expire_flag), icustays (ICU within 24 h) | ~546k admissions | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-CXR-JPG v2.1.0 (metadata + CheXpert labels only) | StudyDate/StudyTime as acquisition time and CheXpert positive-finding counts for CXR severity | ~377k images / 227k studies | PhysioNet credentialed | https://physionet.org/content/mimic-cxr-jpg/2.1.0/ |
| MIMIC-IV-Ext-PE | PE labels for CT pulmonary angiography (severity sensitivity for CTA chest) | subset | PhysioNet credentialed | https://physionet.org/content/mimic-iv-ext-pe/1.0.0/ |
| MIMIC-IV-ED Demo + MIMIC-IV Demo v2.2 | Dry run of ED/hospital joins | 100 patients | Open | https://physionet.org/content/mimic-iv-ed-demo/2.2/ |

No note demo exists; `scripts/download_data.py --synthetic` writes a schema-matching synthetic `radiology` / `radiology_detail` / `edstays` set so the linkage code can be exercised without credentials.

## Methods

Pipeline (modules in `src/rad_delay/`):

1. **Exam table** (`linkage.py`). Join `radiology` with `radiology_detail` pivoted to exam_name / exam_code / cpt_code / parent_note_id; classify modality from exam_name (regex: CXR, CT head, CT abdomen/pelvis, CTA chest, CT other, radiograph extremity/spine, ultrasound, MRI, other); flag preliminary/wet-read and communication markers from text; TAT = storetime - charttime (hours; reports with storetime < charttime or TAT > 7 days are quarantined and audited).
2. **Linkage to ED stays** (`linkage.py`). A report belongs to an ED stay if charttime is within [intime, outtime] of a stay of the same subject (if several stays overlap, the nearest intime wins). Derived per report: `post_imaging_dwell_h` = outtime - charttime, `report_after_exit` = storetime > outtime, `report_to_exit_h` = outtime - storetime, `addendum_after_exit` (any AR child with storetime > outtime), `n_exams_in_stay`, `exam_seq`.
3. **Timing and severity features** (`features.py`). Hour of charttime, night (23:00-07:00), weekend, shift-boundary distance, exam sequence, and CXR severity = number of CheXpert findings labelled positive (1.0) for the study whose StudyTime is within +-30 min of charttime; `anchor_year_group` as covariate.
4. **Descriptives and adjusted models** (`analysis.py`). TAT quantiles per modality x hour block; log-linear OLS for dwell (log hours) and logistic for admission / ICU / 72-h return, with covariates (acuity, age, sex, arrival mode, chief-complaint cluster, modality, n_exams, severity, hour, weekend, anchor group) and cluster-robust SEs by subject.
5. **Instrumental variables** (`analysis.py`). Manual 2SLS with instruments = hour-block x modality dummies and weekend, exogenous covariates as above, heteroskedasticity-robust SEs; first-stage F reported; over-identification (Sargan) reported; exclusion restriction discussed and probed with the negative-control outcomes (arrival-to-exam time, pre-imaging wait) which should show no IV effect.
6. **Regression discontinuity in time** (`analysis.py`). Local-linear RDiT on charttime hour at 07:00 (day-shift start) with bandwidths 1-3 h; outcome TAT and dwell; placebo cut-offs at 11:00 and 15:00.
7. **Pending-report analysis** (`analysis.py`). Among discharged stays: prevalence of report_after_exit and addendum_after_exit by modality; adjusted OR for 72-h return and for admission on return; time-from-exit-to-finalisation distribution.

Tools: pandas, DuckDB, statsmodels, scikit-learn (chief-complaint clusters), regex; no external NLP APIs.

## Evaluation & statistics

- Unit: report (RQ1, RQ5) or ED stay with >= 1 exam (RQ2-RQ4; the index exam is the first CT/CXR); cluster-robust SEs and cluster bootstrap by subject.
- Primary pre-specified tests: H2 (CT abdomen/pelvis dwell), H4 (pending-report 72-h return), H5 (night TAT); Holm-corrected. Modality-specific and hour-specific estimates are exploratory with BH-FDR.
- Confounding by prioritisation: sicker patients get faster reads (would bias TAT effects towards benefit). Handled by severity adjustment (CheXpert counts, acuity), IV, and by reporting the sign flip between crude and adjusted estimates as a finding.
- Negative controls (Lipsitch et al., 2010 Epidemiology): arrival-to-exam time and triage temperature as outcomes that TAT cannot causally affect; a permutation of storetime across reports within modality x hour as a null for RQ2/RQ4.
- Data-quality audit: storetime < charttime, identical timestamps (auto-stored), TAT > 7 d, and reports with missing storetime, tabulated by year group and modality before any modelling; sensitivity analyses excluding auto-stored reports.
- Date shifting: per-patient shifts preserve time of day and day of week but not calendar date, so queue/workload instruments across patients are impossible; only within-patient and time-of-day designs are used, stated explicitly.

## Publishable angle

Headline result: "In a large academic ED, the final radiology report arrived after the disposition decision in X% of imaged visits (Y% for overnight chest radiographs); each hour of CT turnaround added Z minutes of post-imaging dwell and raised admission odds at ESI 3; and discharge with a pending report or a post-discharge addendum was associated with higher 72-hour return." This is the first patient-level, open-data quantification of report timing as a safety and flow variable, with methods that other MIMIC users can reuse (the `storetime`-vs-`charttime` audit alone is useful to the benchmark community).

Target venues: Journal of the American College of Radiology; Annals of Emergency Medicine; Radiology: Artificial Intelligence (as a data-quality / timing benchmark note); JAMIA; Emergency Radiology.

Follow-ups: (a) use MIMIC-CXR report text to detect actionable findings and test whether post-exit reports with actionable findings drive the 72-h return signal; (b) simulate ED flow with a queueing model calibrated on these TAT distributions; (c) replicate on another dataset with report timestamps.

## Risks, confounds & mitigations

- **`charttime` semantics** may be exam time, order time or charting time depending on the source system. Mitigation: cross-check against MIMIC-CXR StudyTime for chest radiographs (expected within minutes); report the distribution of the difference; use StudyTime when available.
- **Auto-finalised or batch-stored reports** (storetime = charttime or clustered at fixed times). Mitigation: audit and exclude; report sensitivity.
- **Prioritisation confounding** (severity to fast reads). Mitigation: severity adjustment, IV, negative controls; interpret crude estimates cautiously.
- **Weak or invalid instruments** (hour of day also affects crowding). Mitigation: first-stage F; interaction instruments (hour x modality) with modality main effects controlled; RDiT as a second design; negative-control outcomes.
- **Multiple exams per stay** and inpatient imaging after admission. Mitigation: index exam definition; charttime restricted to [intime, outtime].
- **72-h return under-ascertainment** (returns elsewhere). Mitigation: acknowledge; use admission-on-return as a stricter outcome.
- **Date-shift edge effects** at year boundaries. Mitigation: exclude reports within 24 h of a subject's shifted year boundary if timestamps look inconsistent.

## Milestones

- [ ] Credentialing for MIMIC-IV-Note, MIMIC-IV-ED, MIMIC-IV, MIMIC-CXR-JPG (metadata/labels).
- [ ] Synthetic dry run (`scripts/download_data.py --synthetic`) and ED/hospital demo joins (`--sample`).
- [ ] Exam table with modality classification validated on 300 hand-checked exam names (>= 95% agreement).
- [ ] Timestamp audit (charttime vs StudyTime; storetime anomalies); linkage to ED stays; RQ1 prevalence tables.
- [ ] Timing/severity features; adjusted models for dwell, admission, ICU, 72-h return (RQ2-RQ4).
- [ ] 2SLS and RDiT with negative controls (RQ2, RQ5); communication moderator (RQ6).
- [ ] Sensitivity analyses (auto-stored reports, index-exam definitions, severity sources).
- [ ] Pre-registration, manuscript, code release without data.

## Ethics / data-use notes

- All MIMIC resources are PhysioNet credentialed (CITI + DUA); data are never redistributed; `data/` is git-ignored; credentials via `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`.
- Report text stays local; regex-only markers, no third-party LLM or cloud APIs (PhysioNet responsible-use policy).
- The study concerns system performance, not individual radiologists or clinicians; no attempt is made to identify readers, and results are reported at modality/time-block level with cells >= 11.
