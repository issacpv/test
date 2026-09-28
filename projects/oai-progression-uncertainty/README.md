# Knee osteoarthritis progression in the OAI with decomposed uncertainty and competing risks: does model uncertainty track reader disagreement, and does it transport to MOST?

One-sentence pitch: Train radiograph + MRI + clinical models for structural knee-OA progression on the Osteoarthritis Initiative (OAI), but evaluate them the way the outcome actually behaves - as a competing-risks time-to-event problem with noisy grades - and decompose predictive uncertainty into label (reader) disagreement, aleatoric and epistemic parts, with conformal guarantees, subgroup audits and external validation on MOST.

## Status / difficulty / timeline / compute

- Status: proposal + starter code (OAI knee-level cohort construction and competing-risk event coding, participant-grouped folds, uncertainty decomposition and Mondrian conformal sets, Aalen-Johansen / cause-specific concordance / IPCW Brier, subgroup metrics).
- Difficulty: PhD-level (imaging ML + biostatistics); 9-12 months including data access.
- Compute: one 24-48 GB GPU for radiograph (2D) and MRI (3D, DESS/IW-TSE) encoders; OAI images are ~20+ TB in total but the analysis needs only bilateral fixed-flexion knee radiographs (~50 GB) and sagittal DESS/IW-TSE for a subset (~1-2 TB). CPU suffices for the statistics.
- Related projects (kept separate): `lidc-nodule-uncertainty` (uncertainty in imaging), `brain-age-transportability` (external validation design), `connectome-prediction-fairness` (subgroup audits).

## Background

The OAI is a multi-centre, 10-year observational study of 4,796 participants with knee radiographs, 3T MRI, biospecimens and patient-reported outcomes (NIH/NIAMS; archive at NDA). Kellgren-Lawrence (KL) grades (Kellgren & Lawrence, 1957) and semi-quantitative MRI scores (MOAKS; Hunter et al., 2011, Osteoarthritis Cartilage) define structural progression; total knee replacement (TKR) is the hard endpoint. Deep learning on OAI radiographs predicts KL grade (Tiulpin et al., 2018, Sci Rep), progression (Tiulpin et al., 2019, Sci Rep; Schiratti et al., 2021, Arthritis Res Ther for MRI) and TKR (Leung et al., 2020, Radiology; Rajamohan et al., 2023, Sci Rep), and MRI-based TKR models have been pushed toward better generalisation (arXiv 2504.19203, 2025). Uncertainty-aware knee radiograph models with conformal prediction and out-of-domain detection appeared in 2024 (PubMed 39477040), and 2026 work reports Mondrian split-conformal prediction for MOAKS-derived cartilage/meniscus targets with expected calibration errors around 0.09-0.12 (Frontiers in Physiology, 2026). MRI imaging biomarkers reach external-validation AUCs of roughly 0.69 (radiographic) and 0.77 (symptomatic) progression.

## The research gap

What has been done:

- Progression treated as a binary label at a fixed horizon (24/48/72 months), ignoring TKR and death as competing events and treating dropout as missing-at-random.
- Uncertainty reported, when at all, as a single scalar (entropy, ensemble variance) or as conformal sets for the KL classification task, not for progression.
- Grade noise is acknowledged (KL inter-reader kappa ~0.5-0.7) but not modelled; the OAI central readings include reader identifiers and repeat/adjudicated reads for reliability subsets that have not been used to ask whether model uncertainty concentrates where readers disagree.
- External validation is rare; MOST (Multicenter Osteoarthritis Study) shares the imaging protocol family and central reading conventions and is the natural external cohort.

What is specifically missing:

1. A competing-risks formulation of structural progression (KL/JSN increase vs TKR vs death vs loss to follow-up) with cause-specific and subdistribution metrics and conformalised time-to-event guarantees (Candes, Lei & Ren, 2023, JRSS-B).
2. Decomposition of uncertainty into reader disagreement (label noise), aleatoric and epistemic (Depeweg et al., 2018, ICML style mutual information from deep ensembles; Lakshminarayanan et al., 2017, NeurIPS), and the test of whether epistemic uncertainty predicts reader disagreement (a novel validity check for medical imaging uncertainty).
3. Subgroup calibration and conformal coverage by sex, race, BMI class and OAI clinical site, and transport of both point predictions and uncertainty to MOST.

Sharpened angle: "Uncertainty that means something": a progression model whose uncertainty (a) is calibrated under competing risks, (b) tracks human reader disagreement, (c) holds coverage across subgroups and (d) survives transport to MOST.

## Research questions / hypotheses

1. H1 (competing risks): Cause-specific models for KL progression that account for TKR/death as competing events give different risk rankings than binary 48-month classifiers (rank correlation < 0.9) and better IPCW Brier scores at 48 and 96 months.
2. H2 (reader disagreement): Epistemic uncertainty (ensemble mutual information) at baseline predicts knee-visits where two OAI readers disagreed on KL grade with AUROC >= 0.65; aleatoric uncertainty does not add (nested logistic regression).
3. H3 (conformal validity): Mondrian (class-conditional) conformal sets achieve nominal 90% coverage overall and within each sex x race x BMI stratum in OAI hold-out (coverage difference across strata < 5 percentage points).
4. H4 (transport): On MOST, discrimination drops by < 0.05 AUROC/C-index, but conformal coverage drops below nominal unless recalibrated on a small MOST calibration set (n = 300); we quantify the calibration-set size needed to restore coverage.
5. H5 (value of MRI): Adding baseline MRI (DESS-derived cartilage thickness or a 3D encoder) improves cause-specific C-index for progression by >= 0.03 over radiograph + clinical, but not for TKR.

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| OAI (clinical: AllClinicalXX, Enrollees, Outcomes99; x-ray readings: kXR_SQ_BUXX; MRI readings: kMRI_*; images: bilateral PA fixed-flexion radiographs, 3T knee MRI) | Training and internal validation | 4,796 participants; >431k visits; ~26.6M images | Free registration on NDA (RAS login via eRA Commons / Login.gov / PIV) + acceptance of OAI data access terms ("NDA-lite"; lighter than the full NDA DUC; verify current terms for image packages) | https://nda.nih.gov/oai |
| MOST (Multicenter Osteoarthritis Study) | External validation | ~3,000 participants, 84-month follow-up + later visits | Application to MOST (free, project proposal + agreement) | https://most.ucsf.edu |
| OAI-ZIB segmentations (Ambellan et al., 2019, Med Image Anal) | Cartilage/bone masks for 507 OAI DESS MRIs | 507 MRIs | Open (segmentations); images via OAI | https://pubdata.zib.de |
| Knee OA severity grading dataset (Chen, 2018, Mendeley Data; OAI-derived radiograph crops with KL labels) | Open smoke test for the radiograph encoder | ~8k crops | Open (CC BY) | https://data.mendeley.com (search title) |

## Methods

1. Cohort (`oai_prog.cohort`): knee-level table from OAI clinical files; imaging visits V00 (0), V01 (12), V03 (24), V05 (36), V06 (48), V08 (72), V10 (96 months); baseline KL 0-3 (exclude KL 4 and prior TKR). Events: 1 = KL increase >= 1 (or medial JSN increase; first visit at which observed), 2 = TKR (Outcomes99), 3 = death; censoring at last completed imaging visit. Participant-grouped folds so both knees stay together.
2. Inputs: baseline radiograph (knee crop, standard pre-processing as in Tiulpin et al., 2018), clinical (age, sex, BMI, WOMAC pain, KL0, prior injury, site), optional MRI (cartilage thickness from OAI-ZIB masks or a 3D encoder on DESS).
3. Models: deep ensemble (M = 5) of ResNet/ConvNeXt encoders with a discrete-time competing-risk head (cause-specific hazards per 12-month interval, DeepHit-style; Lee et al., 2018, AAAI) plus a gradient-boosting clinical baseline; Fine-Gray / cause-specific Cox on tabular features as classical reference.
4. Uncertainty (`oai_prog.uncertainty`): ensemble decomposition (total predictive entropy = expected entropy [aleatoric] + mutual information [epistemic]); Mondrian split-conformal sets for the KL-change class and conformalised lower bounds on event-free time; reader disagreement from repeat/adjudicated central readings.
5. Evaluation (`oai_prog.survival`, `oai_prog.uncertainty`): Aalen-Johansen cumulative incidence, cause-specific C-index, IPCW Brier at 48/96 months, calibration by decile, ECE, coverage/size of conformal sets, subgroup metrics.
6. External validation: freeze models, apply to MOST baseline radiographs; recalibrate conformal thresholds on n = 100/300/1000 MOST knees and report coverage curves.
7. Tools: PyTorch, torchvision/timm, lifelines/scikit-survival, pydicom, SimpleITK, pandas, statsmodels.

## Evaluation & statistics

- Splits: 5-fold participant-grouped CV in OAI for development; one locked hold-out fold for reporting; MOST used once.
- Metrics with 95% CIs via participant-level bootstrap (2,000 resamples); model comparisons by paired bootstrap of differences.
- Multiple comparisons: five pre-registered hypotheses; subgroup tests Holm-corrected within H3.
- Nulls: permutation of reader-disagreement labels for H2; label-shuffled models for a chance C-index; coverage under random conformal scores.
- Leakage: knee-level rows grouped by participant; no follow-up variables in inputs; image pre-processing statistics fitted on training folds only; MOST never touched before the final run.
- Missing follow-up treated by censoring (not exclusion); sensitivity analysis with inverse-probability-of-dropout weights.

## Publishable angle

Headline: "Epistemic uncertainty of a knee-OA progression model concentrates where expert readers disagree, and competing-risk-aware conformal risk transports to MOST after small-sample recalibration."

Target venues: Radiology: Artificial Intelligence; Osteoarthritis and Cartilage (or OA&C Open); Medical Image Analysis; MICCAI/MIDL (methods paper on uncertainty decomposition vs reader disagreement).

Follow-ups: symptomatic progression (WOMAC) as an additional event type; longitudinal (multi-visit) inputs with change maps; fairness-aware recalibration; release of a de-identified per-knee risk table through NDA as a derived dataset.

## Risks, confounds & mitigations

- OAI image access: NDA download of image packages (`downloadcmd`) is slow for TB volumes; request only the needed series; start with radiographs.
- Reader-disagreement subsets are limited in size: pool KL, JSN and osteophyte grades; use adjudicated-vs-initial differences as additional disagreement signal; report power.
- Site/scanner effects: include site as covariate and as a subgroup; MOST has different sites, which is the point of H4.
- Immortal-time and interval censoring: events observed only at visits; use discrete-time hazards aligned to visit schedule.
- Class imbalance (progression ~15-25% at 48 months): report AUPRC and calibration, not only AUROC.
- Race categories are coarse in OAI; report subgroup results as audits, not as biological claims.

## Milestones

- [ ] NDA registration; download OAI clinical files and radiograph packages; MOST application submitted.
- [ ] Knee-level cohort with competing-risk event coding; descriptive cumulative incidence (Aalen-Johansen).
- [ ] Clinical-only baselines (Fine-Gray, gradient boosting); radiograph ensemble with competing-risk head.
- [ ] Uncertainty decomposition, reader-disagreement analysis (H2), Mondrian conformal sets and subgroup coverage (H3).
- [ ] MRI branch (H5); locked hold-out evaluation (H1).
- [ ] MOST external validation and recalibration curves (H4); manuscript.

## Ethics / data-use notes

- OAI data are distributed by NDA under OAI data access terms; MOST under its own agreement. Do not redistribute participant-level data or images; do not upload images or clinical rows to third-party LLM/API services.
- Radiographs and MRIs are potentially re-identifiable in combination with clinical data; keep on approved storage, never commit data (`data/` is git-ignored).
- Report subgroup performance transparently; avoid deploying thresholds tuned on one subgroup.
