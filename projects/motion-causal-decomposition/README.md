# motion-causal-decomposition — How much of "functional connectivity predicts cognition" is head motion? A causal decomposition with negative controls and within-subject instruments

Instead of "controlling for" head motion, treat it as an exposure and decompose the observed FC → cognition association into (i) a trait-confounding path (people who move more differ in cognition for reasons unrelated to the BOLD signal), (ii) an artifact path (motion corrupts FC, and the corrupted edges carry motion information that predicts cognition), and (iii) motion-independent brain signal, using negative-control exposures (motion measured in acquisitions that cannot corrupt the FC being used), within-subject instrumental variables (run order), motion-matched permutation nulls and formal sensitivity bounds, across HCP, ABCD and ABIDE.

## Status / difficulty / timeline / compute

- Status: design + starter code (FD computation, FC features, grouped cross-validated ridge/CPM with per-run predictions, within-subject artifact-sensitivity estimator with 2SLS option, covariance decomposition, negative-control-exposure contrast, motion-matched permutation null, robustness values, generative simulator).
- Difficulty: MSc to PhD chapter; the HCP + ABIDE arm is 6-9 months (both are open/registration), ABCD adds 3-6 months for the application and re-processing.
- Compute: HCP node time series (HCP_PTN1200 release) and ABIDE preprocessed ROI series (PCP) are small (GBs); ridge/CPM with permutations runs on a workstation in hours. ABCD run-level connectomes (ABCD-BIDS derivatives on NDA) are tens of GB; no GPU needed.

## Background

In-scanner head motion produces distance-dependent artefacts in resting-state functional connectivity (Power et al., 2012, NeuroImage; Van Dijk et al., 2012, NeuroImage; Satterthwaite et al., 2012, NeuroImage) that survive standard denoising to a variable degree (Ciric et al., 2017, NeuroImage; Parkes et al., 2018, NeuroImage). Motion is also a stable, heritable trait (Couvy-Duchesne et al., 2014, NeuroImage; Hodgson et al., 2017, Journal of Neuroscience; Zeng et al., 2014, PNAS) that correlates with age, ADHD symptoms, IQ, BMI and many other phenotypes. Consequently "FC predicts cognition" results are always at risk of being partly "motion predicts cognition". The standard responses are censoring, regression of motion summaries, or reporting the motion-phenotype correlation, none of which says *how much* of the association a motion path explains.

Causal epidemiology has tools for exactly this: negative-control exposures and outcomes to detect and quantify confounding/bias (Lipsitch, Tchetgen Tchetgen and Cohen, 2010, Epidemiology), instrumental variables (Angrist, Imbens and Rubin, 1996, JASA), and sensitivity analyses that bound how strong an unmeasured confounder would have to be (Cinelli and Hazlett, 2020, JRSS-B; VanderWeele and Ding, 2017, Annals of Internal Medicine). They have not been used for motion.

## The research gap

What has been done:

- Siegel et al., 2017, Cerebral Cortex showed that data quality (motion) changes which FC-behaviour links are observed in HCP, using stricter censoring and motion-matched subsamples.
- The SHAMAN motion impact score ("Motion impact score for detecting spurious brain-behavior associations", Nature Communications, 2025, 16:8614) assigns each trait-FC relationship an over-/under-estimation score from split-half analysis of high- and low-motion halves in ABCD (n = 7,270; 45 traits; ~40 % of traits with significant over- or underestimation after denoising without censoring). This is a detection tool at the trait level, not a decomposition of the prediction into causal paths, and it is descriptive of association differences between halves.
- ABCD analyses in 2024-2025 ("Balancing data quality and bias..."; "Data rescue in high-motion youth cohorts...", bioRxiv 2024) quantify how exclusion thresholds trade bias against sample representativeness and report that most evaluated phenotypes correlate with motion.
- Denoising benchmarks (Ciric 2017; Parkes 2018) rank pipelines by QC-FC correlations and distance dependence, i.e. by artifact removal, not by what remains in brain-behaviour prediction.
- Twin studies (Couvy-Duchesne 2014; Hodgson 2017) establish motion heritability and shared genetic variance with some traits, which is the trait-confounding path, but do not connect it to FC-based prediction.

What is specifically missing:

1. No decomposition of the FC → cognition prediction into an artifact-mediated component and a trait-confounding component with identifying assumptions stated and tested. Every existing approach either removes motion (and thus removes the trait signal that co-varies with cognition) or reports a residual correlation.
2. No use of *negative-control exposures*: motion measured in a different acquisition (the diffusion scan, the T1 QC metric, or the other day's resting run) shares the trait but cannot mechanically corrupt the FC used for prediction; the difference between the effect of "same-run motion" and "other-acquisition motion" on the prediction isolates the artifact path.
3. No within-subject instrument. HCP has four resting runs over two days and ABCD four runs over two sessions; run order and session shift motion (fatigue) for reasons that cannot change a subject's cognition, which identifies the artifact sensitivity dŷ/dmotion holding the subject fixed.
4. No sensitivity bounds (robustness values, E-values) for the claim "the FC-cognition association survives motion", and no statement of how strong residual motion confounding would need to be to explain a given prediction accuracy.
5. No transfer of the decomposition across cohorts with different motion regimes: HCP (low-motion adults), ABCD (high-motion children), ABIDE (case-control with differential motion), which is where the method's practical value lies.

## Research questions / hypotheses

1. RQ1 (artifact sensitivity). Holding the subject fixed, how much does a subject's predicted cognition change per 0.1 mm of mean framewise displacement (dŷ/dm), estimated from run-to-run variation with run order/session as instrument? H1: dŷ/dm is negative (higher motion → lower predicted cognition) and remains non-zero after global-signal regression and scrubbing, with magnitude ordered ABCD > ABIDE > HCP.
2. RQ2 (decomposition). What fraction of Cov(ŷ, cognition) is artifact-mediated (dŷ/dm × Cov(m, cognition)) versus trait-confounded versus clean? H2: for fluid cognition in HCP, artifact-mediated 10-25 % and trait-confounded 10-20 % under standard denoising; the clean component remains significant; for ABCD, artifact + trait exceed 50 %.
3. RQ3 (negative-control exposure). Does motion during the diffusion scan (or the other day's rest run) predict ŷ as strongly as same-run motion? H3: the other-acquisition coefficient is 40-70 % of the same-run coefficient; the remainder is the artifact path and matches the IV estimate within CI.
4. RQ4 (edge-level). Which edges carry the artifact path (distance-dependent, short-range) versus the trait path (distributed, heritable) versus the clean path? H4: artifact-carrying edges are short-range and their weights flip sign under scrubbing; clean-path edges are stable across pipelines.
5. RQ5 (case-control). In ABIDE, what share of the ASD vs control FC difference and of the FC → symptom-severity prediction is motion-mediated? H5: > 30 % of the group-difference effect size and most of the short-range hyper-connectivity.
6. RQ6 (robustness). What is the robustness value of the clean component? H6: an unmeasured confounder would need to explain > 5 % of residual variance of both ŷ and cognition to nullify it in HCP; in ABCD it is < 2 % for several phenotypes.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| HCP Young Adult S1200 | Four 15-min resting runs (2 days), `Movement_Regressors.txt` / `Movement_RelativeRMS_mean.txt` per run, HCP_PTN1200 node time series (ICA-d100/d200 and Schaefer parcels from the CIFTI dense series), NIH Toolbox cognition (`CogFluidComp_Unadj`, `CogTotalComp_Unadj`), diffusion-scan motion from `eddy` movement logs, twin/family structure (restricted) | ~1,000 subjects with 4 runs; 45 retest | Free registration + open-access data-use terms; family structure and exact age need the Restricted Data application | https://db.humanconnectome.org , https://www.humanconnectome.org/study/hcp-young-adult |
| ABIDE I/II (Preprocessed Connectomes Project) | Preprocessed ROI time series (CC200 / AAL / Craddock), `func_mean_fd` and other QC fields in the phenotypic file, diagnosis, ADOS/SRS, FIQ, age, site | ~1,100 (ABIDE I) + ~1,100 (ABIDE II) | Open (anonymous HTTPS/S3 downloads; ABIDE II needs a free NITRC/INDI registration for some sites) | http://preprocessed-connectomes-project.org/abide/ , http://fcon_1000.projects.nitrc.org/indi/abide/ |
| ABCD Study (release 5.x) | Four resting runs over two sessions, run-level FD, ABCD-BIDS/DCAN connectomes, NIH Toolbox, CBCL, site and scanner (including real-time motion monitoring availability) | ~11,800 children, ~7,000 with usable rest | Application via NIH NDA (Data Use Certification) | https://nda.nih.gov/abcd , https://abcdstudy.org |
| CoRR (Consortium for Reliability and Reproducibility) | Test-retest resting sessions for run-level artifact sensitivity without phenotypes | ~1,600 subjects, multiple sessions | Open (NITRC/INDI) | http://fcon_1000.projects.nitrc.org/indi/CoRR/ |

## Methods

1. Motion and FC features (`motion_causal.fc_features`): Power framewise displacement from six rigid-body parameters (HCP `Movement_Regressors.txt` layout; 50 mm head radius), mean FD and proportion of frames > 0.2 mm per run; Fisher-z Pearson FC per run from node time series; upper-triangle vectorisation; optional scrubbed and GSR variants produced by the same functions on different inputs.
2. Prediction with per-run outputs (`motion_causal.predict`): ridge regression (inner-CV alpha) and CPM (Shen et al., 2017, Nature Protocols) in family-grouped outer folds (twins in the same fold). The model is trained on subject-mean FC of training subjects and then applied to *each run* of held-out subjects, giving ŷ_ir per run; the subject-level prediction is the mean over runs. Haufe-transformed weights (Haufe et al., 2014, NeuroImage) give interpretable edge maps.
3. Artifact sensitivity (`motion_causal.decomposition.artifact_sensitivity`): within-subject regression of ŷ_ir on run mean FD m_ir with subject fixed effects and cluster-robust SEs; 2SLS variant with run order / session as instrument (relevance checked by the first-stage F; exclusion is plausible because run order cannot change cognition, but it may change arousal, hence the drowsiness covariate from HCP's eye-tracking flag where available).
4. Decomposition (`decompose_association`): Cov(ŷ_i, y_i) = Cov(ŷ_i − β_m m_i, y_i) + β_m Cov(m_i, y_i). The second term is the artifact-mediated component; the first is further split into a trait-confounding component (the part of Cov predicted by the negative-control exposure) and the clean remainder. Reported as covariances, correlations and shares with subject-bootstrap CIs.
5. Negative-control exposure (`negative_control_exposure`): regress ŷ_i on same-run motion and on other-acquisition motion (dMRI `eddy` RMS movement; other-day rest FD); report both coefficients, their difference (artifact path) and agreement with the IV estimate.
6. Nulls (`motion_causal.nulls`): motion-matched permutation (phenotype permuted within motion-quantile strata) gives the prediction accuracy achievable from motion alone; standard permutation gives the chance level; the gap is a second estimate of motion-carried accuracy. Sensitivity: Cinelli-Hazlett robustness values for the clean component and E-values for the case-control contrast.
7. Simulation (`motion_causal.simulate`): generative model with latent brain signal, trait motion, run-level motion with an instrument, artifact edges and trait confounding; used to check identification and bias of every estimator before touching real data.

Libraries: numpy, pandas, scipy, scikit-learn, statsmodels; optional nibabel (CIFTI), nilearn (parcellations), `linearmodels` (IV cross-check).

## Evaluation and statistics

- Primary estimands: β_m (dŷ/dm) with cluster-robust and IV CIs; artifact, trait and clean shares of Cov(ŷ, y) with 2,000 subject-level bootstrap resamples (families resampled as units in HCP; sites as strata in ABIDE/ABCD).
- Identification checks: first-stage F > 10 for the instrument; placebo instrument (random run labels) gives β_m ≈ 0; negative-control outcome (a phenotype with no plausible brain cause but a motion correlation, e.g. a synthetic phenotype constructed as motion + noise) yields a clean share ≈ 0.
- Multiverse over denoising: 24p + aCompCor, + GSR, + scrubbing at FD 0.2/0.5 mm; results reported per pipeline and summarised by specification curve.
- Leakage prevention: all model fitting inside outer folds; motion strata for the permutation null computed on the full sample only from motion, never from the phenotype; family-grouped folds; site-stratified folds in ABIDE/ABCD.
- Multiple comparisons: phenotypes are pre-specified (HCP: fluid, crystallised, total cognition, processing speed; ABCD: NIH Toolbox composite, CBCL attention; ABIDE: ADOS total, SRS); edge-level maps use network-level aggregation and FDR.
- Nulls: label permutation (chance), motion-matched permutation (motion-only accuracy), placebo instrument, and simulated data with zero artifact (expect artifact share ≈ 0, coverage ≥ 93 %).

## Publishable angle

Headline: "X % of the FC-cognition prediction in HCP, Y % in ABCD and Z % of the ASD-control FC difference in ABIDE are attributable to motion through artifact and trait paths; the clean component survives with a robustness value of R." The deliverable is a decomposition report that any FC-prediction paper can add (one table: total r, artifact share, trait share, clean share, robustness value), implemented in `motion_causal`.

Target venues: Nature Communications or PLOS Biology (methods + three cohorts), NeuroImage, Imaging Neuroscience, Biological Psychiatry: CNNI (ABIDE/ABCD arm).

Follow-ups: extend to task fMRI activation-behaviour maps; to structural connectivity (eddy motion); to brain-age; and to the fairness setting in `connectome-prediction-fairness` (motion differs by demographic group, so part of the "fairness gap" may be a motion gap).

## Risks, confounds and mitigations

- Exclusion restriction of run order may fail through arousal/drowsiness: include drowsiness proxies where available (HCP eye-tracker flags, ABCD run-level QC), and triangulate with the negative-control-exposure estimate, which does not rely on the instrument.
- Denoising interacts with motion non-linearly: report per-pipeline results and a specification curve rather than one number.
- ABCD site effects and real-time motion feedback differ by scanner: site-stratified folds and a site-level instrument as a sensitivity analysis only.
- Motion proxies from other acquisitions are noisier (attenuation): errors-in-variables correction using the test-retest reliability of each motion measure (HCP retest subset).
- Prediction accuracies are small (r ≈ 0.1-0.3): shares have wide CIs; pre-register phenotypes, use the full HCP sample and report absolute covariances alongside shares.
- Twin structure and family confounding in HCP: family-grouped folds and, as an extension, a within-family (co-twin control) decomposition.

## Milestones

- [ ] Registration: HCP open access (+ restricted data application for family structure); ABIDE downloads; ABCD NDA application.
- [ ] Simulation study of identification and bias (all estimators); pre-registration.
- [ ] HCP: run-level FC, FD, per-run predictions; RQ1-RQ3 for four phenotypes under four pipelines.
- [ ] ABIDE: case-control decomposition (RQ5) with site-stratified folds.
- [ ] ABCD: replication in a high-motion cohort; site-level instrument sensitivity.
- [ ] Edge-level maps (RQ4), robustness values (RQ6), specification curves.
- [ ] Package release with a "decomposition table" API; write-up.

## Cohort definitions and key variables

| Cohort | Definition | Used for |
|---|---|---|
| HCP primary | S1200 subjects with all four resting runs (≥ 1,000 frames after scrubbing at FD 0.2 mm per run), NIH Toolbox scores, no MRI QC issue flag | RQ1-RQ4, RQ6 |
| HCP negative-control | subset with usable diffusion `eddy` movement logs (control exposure) | RQ3 |
| HCP retest | 45 subjects with a second visit | reliability of motion measures (errors-in-variables) |
| ABIDE case-control | ASD and typical controls, age 6-30, `func_mean_fd` available, sites with ≥ 20 subjects | RQ5 |
| ABCD | baseline + 2-year rest with ≥ 2 usable runs per session, site and scanner recorded | replication, high-motion regime |

Run-level variables: `subject`, `run`, `session`, `run_order` (instrument), `mean_fd`, `frac_fd_gt_0.2`, `fc_vector` (edges), optional `drowsiness_flag`; subject-level: phenotype (`CogFluidComp_Unadj`, `CogTotalComp_Unadj`, `ProcSpeed_Unadj`, ABIDE `ADOS_TOTAL`/`SRS_RAW_TOTAL`/`FIQ`, ABCD NIH Toolbox composite, CBCL attention), `control_motion` (dMRI RMS movement; other-day rest FD), `family_id`, `site`, age, sex.

## Starter code map

| Module / function | What it does |
|---|---|
| `motion_causal.fc_features.framewise_displacement`, `load_hcp_movement_regressors`, `motion_summary` | Power FD from rigid-body parameters (HCP file layout), run summaries |
| `fc_features.fisher_z_fc`, `vectorize_upper`, `edge_distances` | FC features and edge distances (for the distance-dependence check) |
| `motion_causal.predict.crossval_run_predictions` | family-grouped nested CV (ridge or CPM) returning *per-run* out-of-fold predictions and Haufe patterns |
| `motion_causal.decomposition.artifact_sensitivity` | within-subject dŷ/dm with fixed effects or 2SLS (run order as instrument; first-stage F) |
| `decomposition.decompose_association` | Cov(ŷ, y) → artifact-mediated, trait-confounded, clean components and shares |
| `decomposition.negative_control_exposure`, `robustness_value`, `e_value` | negative-control contrast and sensitivity bounds |
| `motion_causal.nulls.motion_matched_permutation`, `permutation_pvalue`, `bootstrap_ci` | motion-only null, p-values, family-block bootstrap |
| `motion_causal.simulate.simulate_motion_cohort` | generative model with known artifact and trait paths |
| `tests/test_motion_causal.py` | FD arithmetic, FE/IV recovery of a known sensitivity, decomposition against ground truth, zero-artifact null |

Quick start:

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/download_data.py abide --out data/abide --sample 20   # open data, no credentials
```

## Ethics / data-use notes

- HCP: cite the WU-Minn HCP consortium; restricted data (family IDs, exact ages) may not be redistributed or combined with public identifiers.
- ABCD: NDA Data Use Certification; derived subject-level tables stay off git; publications require the ABCD acknowledgement text.
- ABIDE: open, but site-level phenotypic files should not be re-hosted; cite each site's contributors as required by INDI.
- No data are sent to third-party APIs. `data/` and `outputs/` are git-ignored.
