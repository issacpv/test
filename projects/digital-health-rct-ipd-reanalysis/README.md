# digital-health-rct-ipd-reanalysis: engagement is not a dose - a pre-registered IPD re-analysis programme for digital-health RCTs

**One-sentence pitch.** Build a registry of completed digital-health RCTs (apps, digital therapeutics, remote monitoring, internet-delivered CBT) from their ClinicalTrials.gov IPD-sharing statements, obtain individual participant data through Vivli / YODA / CSDR / the NIMH Data Archive and open repositories, and re-analyse them under one pre-registered protocol that (i) estimates causal engagement effects with randomisation-respecting methods instead of within-arm correlations, (ii) tests whether reported effects survive informative attrition (reference-based and delta-adjusted multiple imputation), (iii) evaluates heterogeneity of treatment effect with internal-external cross-validation across trials, and (iv) quantifies how much digital-trial evidence is actually retrievable.

## Status / difficulty / timeline / compute

- Status: design + working starter code (this repo): ClinicalTrials.gov v2 registry client, ITT / naive per-protocol / Wald-CACE / principal-score stratified engagement estimators with cluster bootstrap, MAR / jump-to-reference / delta-adjusted multiple imputation with Rubin's rules and tipping-point scans, an interaction-model HTE with internal-external CV, and a multi-trial IPD simulator that encodes engagement confounding and MNAR dropout. All tested.
- Difficulty: MSc to PhD depending on how many platforms are pursued. The registry study alone (RQ1) is an MSc project with no data agreements; the full programme is PhD-scale because data access takes 6-12 months per platform.
- Timeline: registry 2 months; applications submitted at month 2-3; data access months 6-12; analyses run inside secure environments (Vivli/YODA/CSDR) months 9-15; writing in parallel. Open-repository IPD can be analysed from month 3.
- Compute: laptop. Multiple imputation and bootstraps are seconds to minutes per trial.

## Background

Digital mental-health interventions have the largest evidence base in digital health: 176 RCTs (21,702 participants) of mental-health apps show small-to-medium pooled effects on depression and anxiety with considerable heterogeneity (Linardon et al., 2024, *World Psychiatry*), and an updated 2025 meta-analysis of standalone apps (*Lancet Digital Health*) reaches similar conclusions. Attrition is the defining problem of the field ("the law of attrition", Eysenbach, 2005, *JMIR*; Pratap et al., 2020, *npj Digit. Med.*): a 2025 meta-analysis of 79 depression-app trials reports high uptake but substantial dropout, and real-world retention is far lower than in trials (Baumel et al., 2019, *JMIR*). Engagement is widely assumed to be the "dose": a 2025 meta-analysis of 13 studies finds a small positive engagement-outcome association. But engagement is self-selected - participants who engage differ in motivation, severity and circumstances - so within-arm correlations cannot be read causally, and controls typically have no access to the app, which makes the *randomisation* the only clean instrument. Control-app ("digital placebo") effects (Torous & Firth, 2016, *Lancet Psychiatry*) further complicate aggregate synthesis.

Individual participant data solve several of these problems at once, and the infrastructure exists: Vivli (Bierer, Li, Barnes & Sim, 2016, *NEJM*) and the YODA Project (Krumholz & Waldstreicher, 2016, *NEJM*) host thousands of sponsor trials in secure environments; the NIMH Data Archive holds participant-level data from NIMH-funded trials; and ClinicalTrials.gov requires an IPD-sharing statement for every registered trial.

## The research gap

**What has been done (2017-2026):**

- IPD meta-analyses of *internet-delivered CBT* by academic consortia: self-guided iCBT for depression (Karyotaki et al., 2017, *JAMA Psychiatry*; 13 trials), guided vs unguided iCBT IPD network meta-analysis (Karyotaki et al., 2021, *JAMA Psychiatry*), and a component IPD network meta-analysis dismantling iCBT (Furukawa et al., 2021, *Lancet Psychiatry*). These use data pooled by collaborators, not platform IPD, and treat adherence descriptively or as a moderator, not with principal-stratum methods.
- Aggregate meta-analyses of apps (Linardon et al., 2024; *Lancet Digital Health*, 2025), of engagement-outcome associations (2025; 13 studies) and of uptake/attrition (2025; 79 trials).
- Methodology exists but has not been applied at scale to digital trials: principal stratification (Frangakis & Rubin, 2002, *Biometrics*; Jo & Stuart, 2009, *Stat Med*; Ding & Lu, 2017, *JRSS-B*), reference-based MI (Carpenter, Roger & Kenward, 2013, *J Biopharm Stat*; Cro et al., 2020, *Stat Med*), PATH-style HTE (Kent et al., 2020, *Ann Intern Med*), internal-external validation (Steyerberg & Harrell, 2016, *J Clin Epidemiol*; Riley, Tierney & Stewart, 2021, IPD meta-analysis textbook).

**What is missing:**

1. **No one has measured how much digital-trial IPD is retrievable**: the share of completed digital-health RCTs with an IPD-sharing "Yes" statement, where they say it is, and whether it is actually there.
2. **No IPD analysis of platform-hosted digital trials** (industry DTx, remote monitoring, device-app trials on Vivli/YODA), where the pivotal trials and the richest engagement telemetry sit.
3. **Engagement has not been analysed causally across trials.** Engagement-outcome evidence is within-arm and correlational; CACE / principal-score dose-strata estimates and their comparison with the naive numbers are absent.
4. **Attrition sensitivity is not systematically re-run.** Most digital trials report MAR-based analyses; how many conclusions survive jump-to-reference or a plausible delta is unknown.
5. **HTE claims are not transported**: baseline-severity moderation is reported trial by trial, never validated in held-out trials with calibration of predicted benefit.
6. **Early telemetry as a predictor** (first two weeks of use) is unexplored across trials because it exists only in IPD.

## Research questions / hypotheses

1. **RQ1 (retrievability).** *H1:* among completed digital-health RCTs registered since 2015 with >= 100 participants, < 25% state IPD sharing = Yes; industry sponsors are less likely than academic sponsors after adjusting for enrolment and year; fewer than half of "Yes" statements name a repository, and fewer than half of those trials are findable on it.
2. **RQ2 (engagement causal effect).** *H2:* principal-score / CACE engager effects are attenuated by >= 50% relative to naive engagers-vs-controls contrasts; the naive engagers-vs-non-engagers contrast within the treated arm overstates the effect the most.
3. **RQ3 (dose-response).** *H3:* stratifying engagement into tertiles of dose shows a plateau rather than a monotone dose-response after principal-score adjustment.
4. **RQ4 (attrition).** *H4:* under jump-to-reference imputation at least one third of statistically significant ITT effects lose significance; the tipping-point delta is < 0.3 SD for the majority of trials.
5. **RQ5 (HTE).** *H5:* baseline severity is the only moderator with a pooled internal-external calibration slope whose CI excludes 0; machine-learning HTE models (causal forests) do not improve calibrated benefit out of trial.
6. **RQ6 (early telemetry).** *H6:* first-two-week engagement features predict endpoint response within trial (AUROC >= 0.70) but transfer poorly (drop >= 0.10 AUROC) across trials.

## Datasets

| Dataset / source | Used for | Size | Access | URL |
|---|---|---|---|---|
| ClinicalTrials.gov v2 API | Registry of digital-health RCTs and IPD-sharing statements (RQ1) | thousands of trials | Open, no key | https://clinicaltrials.gov/data-api/api |
| Vivli | IPD of sponsor (pharma/device/DTx) and academic trials (RQ2-6) | > 4,000 trials on the platform (subset digital) | Application, DUA, secure environment | https://vivli.org |
| YODA Project | IPD of Johnson & Johnson and partner trials incl. devices | hundreds of trials | Application, DUA, secure environment | https://yoda.yale.edu |
| ClinicalStudyDataRequest.com | Remaining sponsor trials | hundreds | Application, DUA | https://www.clinicalstudydatarequest.com |
| NIMH Data Archive (NDA) | NIMH-funded digital mental-health RCTs, participant level | many collections | Institutional data-access request | https://nda.nih.gov |
| Open IPD (OSF, Zenodo, Dryad, ICPSR) | Trials that shared data with their papers | tens of trials | Open (licence) | per trial |
| Simulated IPD (`dh_ipd.simulate`) | Development, tests, power analysis | any | Generated | this repo |

The exact trial list is an output of RQ1 and is recorded in `data/registry/screening.csv`; it is not fixed a priori.

## Methods

1. **Registry** (`dh_ipd.registry`, `scripts/download_data.py`): v2 API queries per digital-intervention term (`query.intr`), completed interventional trials since 2015, `ipdSharingStatementModule` fields; de-duplication; keyword-based digital flag with manual screening; repository classification from the free-text statement and URL; logistic regression of `ipd_yes` on sponsor class, enrolment, year, condition area; a verification step that searches each named platform for the NCT id.
2. **Harmonisation**: one tidy row per participant (`data/README.md` schema) with a trial-specific, pre-specified engagement definition (binary + continuous dose) and standardised primary outcome; control type recorded.
3. **Engagement effects** (`dh_ipd.engagement`): ANCOVA ITT; naive per-protocol contrasts; Wald/IV CACE with delta-method SE (one-sided non-compliance by design); principal-score weighting with a logistic/multinomial model for engagement strata fitted in the treated arm and applied to controls, ANCOVA-adjusted, with participant and trial (cluster) bootstrap. Dose tertiles for RQ3.
4. **Attrition** (`dh_ipd.missing`): complete case; proper MAR MI; jump-to-reference MI; delta-adjusted MI over a delta grid with the tipping point; Rubin's rules with Barnard-Rubin df. Per trial and pooled (two-stage IPD meta-analysis with random effects).
5. **HTE** (`dh_ipd.hte`): centred interaction model (baseline severity, age, sex, prior treatment) with robust SEs; predicted benefit; internal-external CV with calibration slope/intercept of predicted benefit and quartile checks; exploratory causal forest in the same CV loop.
6. **Early telemetry** (RQ6): first-14-day features (active days, sessions, modules, time-of-day entropy) -> logistic model of response (>= 50% improvement) with trial-wise and leave-one-trial-out AUROC.
7. **Simulation** (`dh_ipd.simulate`): multi-trial generator with unobserved motivation driving engagement and outcome, latent engagement strata, MNAR dropout; used for unit tests, for bias demonstrations and for a power analysis at the realised trial sizes.

Tools: `numpy`, `scipy`, `pandas`, `scikit-learn`, `requests`; `statsmodels` (optional; two-stage random-effects pooling); `econml` (optional; causal forests). Everything must run inside a sponsor's secure environment, hence the minimal dependency set.

## Evaluation & statistics

- Pre-registration (OSF) of the trial-inclusion rules, engagement definitions, the primary estimand for each RQ, and H1-H6 before any IPD is received; the registry analysis is run first and its result does not alter the IPD SAP.
- Estimands: ITT (primary), engager principal-stratum effect (RQ2), dose-stratum effects (RQ3), ITT under J2R / delta (RQ4); all in standardised outcome units with 95% CIs from robust SEs and cluster bootstrap.
- Pooling: two-stage random-effects IPD meta-analysis of trial-level estimates (DerSimonian-Laird and REML), with prediction intervals; one-stage mixed models as sensitivity.
- Multiplicity: each RQ has one primary contrast; secondary contrasts Holm-corrected within RQ.
- Validity checks: principal ignorability assessed by the balance of observed covariates across predicted strata; the exclusion restriction holds by design (controls have no app); MI models include all covariates in the analysis model; MNAR sensitivity is the point, not a nuisance.
- Nulls: (i) permutation of `z` within trial (effects vanish); (ii) simulated data with `engagement_effect = 0` (principal-score estimates ~0 while naive contrasts remain biased); (iii) simulated MNAR with known truth to check which method recovers it.
- Leakage: HTE models never see the held-out trial; telemetry features restricted to the first 14 days; engagement definitions fixed before outcomes are examined.

## Publishable angle

- **Headline.** "Of N completed digital-health RCTs, X% promise IPD and Y% deliver; in the M trials re-analysed, the causal effect of engaging with the app is half the naive estimate, most reported effects survive jump-to-reference but a delta of Z SD tips them, and baseline severity is the only moderator that transports." The registry paper is a stand-alone meta-research result publishable within months.
- **Venues.** *Lancet Digital Health* or *npj Digital Medicine* (IPD re-analysis); *JAMA Network Open* / *BMJ* (registry and retrievability audit); *Journal of Medical Internet Research* (digital-health audience); *Statistics in Medicine* / *Clinical Trials* for the principal-stratum dose methodology paper.
- **Follow-ups.** Extend to remote-monitoring device trials (heart failure, diabetes CGM) on YODA/Vivli; a living registry updated from the API; an engagement-estimand reporting guideline proposal (CONSORT-EHEALTH extension); JITAI micro-randomised trials as the next data type.

## Risks, confounds & mitigations

- **Data access failures** (platform review, sponsor refusal, trials not actually deposited). Mitigation: RQ1 turns this into a measured outcome; open-repository and NDA trials provide a guaranteed analysis set; applications to several platforms in parallel.
- **Engagement definitions differ across trials.** Mitigation: trial-specific binary definitions fixed a priori plus a continuous dose in tertiles; sensitivity to alternative cut-offs reported.
- **Principal ignorability may fail** (unobserved motivation drives both engagement and outcome even given covariates). Mitigation: report CACE (assumption-light) alongside principal-score estimates; sensitivity analysis on the correlation between stratum and potential outcome; the simulator quantifies bias under violation.
- **Control types vary** (waitlist vs attention control). Mitigation: control type as a moderator in pooling; primary analysis stratified.
- **Secure environments restrict software.** Mitigation: the package depends only on numpy/scipy/pandas/scikit-learn/requests and is installed from source.
- **Outcome scales differ** (PHQ-9, BDI, GAD-7). Mitigation: within-trial standardisation; RQ4/RQ5 report per trial before pooling.

## Milestones

- [ ] Registry extract (`scripts/download_data.py`), manual screening of the digital flag, repository verification; RQ1 analysis and pre-print.
- [ ] Pre-register the IPD SAP (OSF); submit Vivli / YODA / CSDR / NDA applications; collect open-repository IPD.
- [ ] Simulation study: bias of naive vs CACE vs principal-score estimators; MI method recovery under MNAR; power at realised sizes.
- [ ] Harmonise the first trials (open + NDA); run RQ2-RQ5 per trial.
- [ ] Platform trials inside secure environments; export aggregate outputs.
- [ ] Two-stage pooling; HTE internal-external CV; telemetry analyses.
- [ ] Manuscripts: registry paper, IPD re-analysis paper, methods paper.

## Ethics / data-use notes

- IPD from Vivli, YODA and CSDR stays inside the providers' secure environments; NDA data stay on approved institutional storage; each DUA's publication review and re-identification prohibitions are followed; only aggregate results (counts >= 10) leave.
- IPD must never be uploaded to third-party services, including LLM APIs; the analysis code is dependency-light so it runs where the data are.
- Open-repository IPD is used under its licence with citation of the original trial; participants consented to secondary use as stated in each dataset's documentation.
- The registry uses only public trial metadata; sponsor names appear in the registry paper as public registry information.
- Never commit IPD, participant-level derived tables or DUA documents; `data/` is git-ignored except this documentation.
