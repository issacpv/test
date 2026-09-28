# fmri-hemodynamic-aging — Blood-arrival lag and HRF-shape maps from resting-state fMRI as vascular-aging biomarkers

Build lifespan normative charts of resting-state hemodynamic timing (systemic low-frequency-oscillation blood-arrival lag and deconvolved HRF shape) in HCP-Aging, validate them against multi-delay ASL arterial transit time in the same scans, and test whether a "hemodynamic age" tracks vascular risk, white-matter hyperintensities and amyloid in OASIS-3 better than chronological age does, while quantifying how much of the classic FC-age effect is hemodynamic rather than neural.

## Status / difficulty / timeline / compute

- Status: design + starter code (lag mapping, point-process HRF estimation, heteroscedastic normative model, hemodynamic-age delta, ICC / partial correlation / mediation).
- Difficulty: strong MSc thesis or first PhD chapter. 9-12 months (3 months for HCP-Aging reliability + ASL validation paper; 6 more for the OASIS-3 vascular-risk / amyloid paper).
- Compute: lag mapping with rapidtide is ~10-20 CPU-minutes per 6-7 min run. HCP-Aging has 4 rest runs per participant (~1,200 participants, ~4,800 runs) → ~1,000-1,500 CPU-hours; OASIS-3 adds ~2,000 runs but needs fMRIPrep first (~3-4 CPU-hours per session). Storage: plan ~1.5-2 TB for HCP-Aging minimally preprocessed rfMRI + ASL; start with a 300-participant age-stratified subset. No GPU needed.

## Background

The BOLD signal is a vascular signal. With age, arterioles stiffen and capillary density falls, blood transit through the brain slows, and the hemodynamic response to neural activity becomes smaller and later (West et al., 2019, NeuroImage; Handwerker et al., 2007, Human Brain Mapping). Two consequences follow. First, hemodynamic timing is itself a candidate biomarker of cerebrovascular health that comes for free from any resting-state scan, without gas challenges or contrast agents. Second, resting-state functional-connectivity (FC) differences attributed to "aging" are partly hemodynamic confounds (Tsvetanov et al., 2015, Human Brain Mapping; Tsvetanov, Henson and Rowe, 2021, Phil Trans R Soc B; Rangaprakash et al., 2018, Magnetic Resonance in Medicine).

Two families of resting-state methods capture hemodynamic timing without a task. (i) Blood-arrival lag mapping: a systemic low-frequency oscillation (sLFO, ~0.01-0.15 Hz) travels with the blood and arrives at each voxel with a delay that can be recovered by cross-correlation against a global or refined reference (Tong, Hocke and Frederick, 2019, Frontiers in Neuroscience; the rapidtide toolbox). Lag maps reproduce perfusion deficits in stroke and Moyamoya (Lv et al., 2013, Annals of Neurology; Amemiya et al., 2014, Radiology; Christen et al., 2015, JMRI) and a venous-drainage change with aging has been reported in a large cohort (Aso et al., 2020, Brain). (ii) Resting-state HRF estimation: blind or point-process deconvolution recovers a per-voxel HRF whose height, time-to-peak and width can be mapped (Wu et al., 2013, Medical Image Analysis; rsHRF toolbox, Wu et al., 2021, NeuroImage).

## The research gap

What has been done (2019-2026):

- Task-based HRF changes with age are established (West et al., 2019, NeuroImage; and two recent whole-cortex analyses, "Global effects of aging on the hemodynamic response function in the human brain", 2023, and a 2024 follow-up, reporting roughly twice as much cortex with age-related amplitude change as with timing change).
- In HCP-Aging, physiological (respiratory and cardiac) fMRI response functions become slower (respiratory) and faster (cardiac) with age, with an inflection after ~60 years ("Functional MRI signatures of autonomic physiology in aging", Communications Biology, 2025), and spectral properties of resting hemodynamics change with age (Park et al., 2025, Advanced Science).
- Lag-mapping methodology continues to mature (an improved delay/strength estimation paper in 2026; learned hemodynamic-coupling inference, arXiv 2601.00973, 2026), but applications remain clinical case series (stroke, Moyamoya, TBI) or single-site aging samples.

What is specifically missing:

1. No cohort-scale validation of rs-fMRI blood-arrival lag against an independent transit-time measurement. HCP-Aging acquired multi-delay pCASL (Harms et al., 2018, NeuroImage) that yields arterial transit time (ATT) in the same session; nobody has reported lag-vs-ATT agreement across ~1,000 people.
2. No normative (centile) charts for hemodynamic timing. Brain charts exist for volume (Bethlehem et al., 2022, Nature) and, in this repository's `lifespan-normative-models`, for other structural measures; nothing equivalent exists for lag dispersion or HRF time-to-peak, so a clinician cannot ask "is this person's hemodynamic timing abnormal for their age?".
3. No joint model of hemodynamic timing, vascular risk factors, WMH volume and amyloid in the same participants. OASIS-3 has rs-fMRI, FLAIR, PiB/AV45 Centiloids, UDS vascular history and CDR, but its rs-fMRI has been used almost only for FC.
4. No explicit decomposition of the FC-age effect into a hemodynamic (lag/HRF) part and a residual part using both correction strategies (sLFO regression and HRF deconvolution) on the same data; existing corrections used calibrated task designs or single methods.
5. Test-retest reliability of voxel- and parcel-level lag and HRF timing in older adults is unreported at scale, although HCP-Aging has two same-visit sessions.

## Research questions / hypotheses

1. RQ1 (reliability). Across HCP-Aging REST1 vs REST2, what is the ICC(2,1) of parcel-level lag, lag-map dispersion (GM interquartile range), HRF time-to-peak and FWHM? H1: ICC ≥ 0.6 for parcel lag and lag dispersion, 0.3-0.5 for HRF time-to-peak; reliability decreases with head motion but not with age after motion adjustment.
2. RQ2 (validity). Do rs-fMRI lag maps agree with pCASL ATT maps within participants (spatial correlation across Schaefer-400 + Tian-S2 parcels) and between participants (mean GM lag vs mean ATT)? H2: within-participant spatial r = 0.3-0.6 (spin-test significant), between-participant r ≈ 0.3; agreement is higher in arterial-border-zone parcels than in venous-dominated parcels.
3. RQ3 (normative aging). Lag dispersion, mean arteriovenous lag gradient and HRF time-to-peak increase non-linearly with age, with steeper slopes after 60, and show sex differences (women later menopause-related acceleration). Centiles are estimated with a heteroscedastic spline model and cross-checked with GAMLSS/PCNtoolkit.
4. RQ4 (vascular risk). In HCP-Aging, the hemodynamic-age delta (predicted minus chronological age from lag/HRF features) is associated with systolic blood pressure, pulse pressure, HbA1c, BMI and smoking after adjusting for age, sex and motion. In OASIS-3, the delta is associated with hypertension history, WMH volume and CDR ≥ 0.5, but only weakly with amyloid Centiloid in cognitively normal participants. H4: WMH volume explains more variance in the delta (partial R² ≥ 0.05) than Centiloid (< 0.02).
5. RQ5 (FC confound). After (a) sLFO regression and (b) HRF deconvolution, the age effect on FC (Schaefer-400 edge-wise standardized slopes) is attenuated by 20-40%, with the largest attenuation in visual and default-mode networks; the hemodynamic-age delta mediates a significant fraction of the age → within-DMN FC effect.
6. RQ6 (transport). Normative centiles fitted on HCP-Aging (Prisma, TR 0.8 s) transfer to OASIS-3 (TIM Trio TR 2.2 s; Biograph mMR) after resampling-aware lag estimation and site harmonization (ComBat-GAM on the overlapping 42-90 age band), with a median absolute z-shift < 0.3 in cognitively normal OASIS-3 participants.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| HCP-Aging (Lifespan HCP, HCP-A; Bookheimer et al., 2019, NeuroImage) | Minimally preprocessed rfMRI (4 runs, TR 0.8 s, 2 mm, multiband 8), physiological recordings (respiratory belt, pulse ox), multi-delay pCASL (CBF, ATT), T1w/T2w, blood pressure, blood assays (HbA1c, lipids), demographics, medications | ~1,200 participants, ages 36-100+ | Free; NDA account + Data Use Certification (NIMH Data Archive); no redistribution | https://nda.nih.gov/ (collection 2847) ; https://www.humanconnectome.org/study/hcp-lifespan-aging |
| OASIS-3 (LaMontagne et al., 2019, medRxiv) | rs-fMRI (2 runs, TR 2.2 s, Trio/mMR), T1w, FLAIR, PiB/AV45 PET Centiloids, UDS clinical data (hypertension, diabetes, CDR, APOE) | ~1,378 participants, ~2,800 MR sessions, ages 42-95 | Free registration + DUA on NITRC / XNAT Central | https://sites.wustl.edu/oasisbrains/ ; https://www.nitrc.org/projects/oasis3/ |
| NKI-Rockland Sample (enhanced; Nooner et al., 2012, Frontiers in Neuroscience) | Optional replication of the age curve: rest fMRI at TR 0.645 s and 1.4 s with physio, ages 6-85 | > 1,000 participants | Free DUA; raw BIDS on AWS S3 (fcp-indi bucket); phenotypes via NKI DUA | http://fcon_1000.projects.nitrc.org/indi/enhanced/ |
| OpenNeuro ds000030 (UCLA CNP; Poldrack et al., 2016, Scientific Data) | Open smoke-test data for the pipeline (rest fMRI, T1w) | 272 participants, ages 21-50 | Open (CC0) | https://openneuro.org/datasets/ds000030 |
| Cam-CAN (Shafto et al., 2014, BMC Neurology) | Optional external replication (rest fMRI, ages 18-88) | ~650 with fMRI | Application to Cam-CAN data access committee | https://www.cam-can.org/ |

## Methods

1. Preprocessing. HCP-Aging: use the HCP minimally preprocessed volume series (motion-corrected, distortion-corrected, MNI) but NOT the ICA-FIX-cleaned series, because FIX removes components that carry the sLFO. Regress 24 motion parameters and censor frames with FD > 0.5 mm (Power). OASIS-3 and ds000030: fMRIPrep 23.x with the same confound regression. Physiological confounds (RVT, heart rate) are estimated with `hemo_aging`-compatible code paths (respiration variation is itself a driver of sLFO, so it is modelled as a covariate in RQ3-RQ5, not regressed out before lag mapping).
2. Lag mapping (`hemo_aging.lag_mapping`, and rapidtide 2.x for production). Band-pass 0.009-0.15 Hz; pass 1 reference = global GM mean; cross-correlate every voxel over -7.5 to +15 s with parabolic sub-sample interpolation; refine the reference by shifting and averaging voxels with r > 0.3; three passes. Outputs: lag (s), max correlation, and sLFO-regressed residual series. Summary features: parcel median lag (Schaefer-400 + Tian S2), GM lag IQR, lag skewness, arterial-venous gradient (median lag of the 10% earliest voxels minus median of the 10% latest), fraction of voxels with r > 0.3.
3. HRF estimation (`hemo_aging.hrf_estimation`; rsHRF for production). Point-process pseudo-event detection at z > 1 with a minimum event separation, FIR estimation over a grid of onset delays, and extraction of height, time-to-peak and FWHM per parcel; Wiener deconvolution produces "neural" series for RQ5.
4. ASL. HCP-Aging multi-PLD pCASL processed with the HCP-ASL pipeline (oxford_asl / BASIL) to CBF and ATT maps; parcel medians; ATT quality flags at long ATT.
5. Vascular covariates. HCP-Aging: SBP/DBP (mean of two readings), pulse pressure, HbA1c, total/HDL cholesterol, BMI, smoking status. OASIS-3: UDS hypertension/diabetes/hypercholesterolemia history, WMH volume from FLAIR (LST-LPA or BIANCA; log-transformed, ICV-normalized), Centiloid from PUP, CDR.
6. Normative model (`hemo_aging.normative`). Cubic B-spline (df = 4) mean model with a spline log-variance model fitted on HCP-Aging by sex; z-scores and centiles; cross-check with PCNtoolkit BLR with SHASH likelihood. Hemodynamic age: ridge regression on ~800 lag/HRF parcel features with 10-fold participant-grouped CV and training-fold bias correction.
7. FC-age decomposition. Schaefer-400 FC computed from (a) confound-regressed, (b) sLFO-regressed, (c) HRF-deconvolved series; edge-wise standardized age slopes; attenuation ratio per network; mediation of age → FC by hemodynamic delta with 2,000 bootstrap resamples (`hemo_aging.stats`).

Libraries: numpy, scipy, pandas, statsmodels, scikit-learn; nibabel/nilearn for images; rapidtide, rsHRF (Python), fMRIPrep, oxford_asl; neuromaps for spin tests; PCNtoolkit (optional).

## Evaluation and statistics

- Reliability: ICC(2,1) with bootstrap CIs; Bland-Altman limits for global measures; reliability vs motion tested with a linear model including age.
- Validity: spatial correlations between lag and ATT parcel maps tested with spin-test nulls (neuromaps; 10,000 rotations) and, for subcortex, Moran spectral randomization; between-participant correlations with 95% bootstrap CIs.
- Aging: generalized additive models (age spline by sex) with motion, RVT and heart rate as covariates; knots and df chosen by AIC; change points tested by comparing segmented vs smooth fits.
- Vascular risk: hierarchical linear models; partial R² per risk factor; Holm correction over the pre-registered set of 6 risk factors in HCP-Aging and 4 in OASIS-3.
- FC: edge-wise age slopes with FDR (q = 0.05) within each pipeline; attenuation ratios summarized per Yeo-7 network with participant-level bootstrap; mediation indirect effect with percentile CIs.
- Leakage prevention: normative and age-prediction models are fitted on HCP-Aging only and applied unchanged to OASIS-3; participant-grouped folds for all CV; harmonization parameters estimated on cognitively normal participants only.
- Nulls: label-shuffled hemodynamic-age models (1,000 permutations) to bound chance performance; time-series phase-randomization null for lag maps to estimate the false-lag floor in each run.

## Publishable angle

Headline result: resting-state blood-arrival lag maps agree with ASL arterial transit time at cohort scale and yield a reliable "hemodynamic age" that tracks blood pressure, HbA1c and WMH burden more strongly than amyloid, while 20-40% of the canonical FC-age effect disappears after hemodynamic correction. That combination (validation + normative chart + vascular-risk association + confound accounting) is a two-paper package.

Target venues: NeuroImage; Imaging Neuroscience; Journal of Cerebral Blood Flow & Metabolism; Human Brain Mapping; Neurobiology of Aging.

Follow-ups: (i) apply the normative charts to ADNI and UK Biobank rs-fMRI (no physio, so the RVT-adjusted variant is needed); (ii) link to the breath-hold CVR charts in `cvr-normative-maps` (lag from rest vs delay from breath-hold in NKI-RS, which has both); (iii) lag maps as a screening marker before TMS/tES targeting (see `tms-target-connectivity`).

## Risks, confounds and mitigations

- sLFO lag mixes arterial arrival with venous drainage; parcel-level interpretation must separate arterial-territory from sinus-adjacent voxels (use a venous mask from T2*/SWI or the Ward/Aso venous atlas) and report both.
- TR and scanner differences (0.8 s vs 2.2 s) change lag precision; we simulate downsampling of HCP-Aging data to TR 2.2 s to estimate the induced bias before pooling with OASIS-3.
- Respiration variation and heart-rate changes generate sLFO-like signals; use physio covariates in HCP-Aging and RVT proxies in OASIS-3 (no physio), and show results with and without.
- Motion increases with age; use censoring, FD covariates and a motion-matched sub-analysis.
- ASL ATT is unreliable at long transit times (older participants, border zones); flag and exclude ATT > 2.2 s parcels in a sensitivity analysis.
- OASIS-3 scanner heterogeneity (Trio vs mMR): include scanner as a covariate and ComBat-GAM within the normative transfer.
- Selection: HCP-Aging excludes major vascular disease, compressing the vascular-risk range; OASIS-3 provides the extension to clinical range.
- Multiple hypotheses: primary outcomes are pre-registered (RQ2 spatial r; RQ4 WMH partial R²; RQ5 attenuation ratio), the rest are exploratory.

## Milestones

- [ ] Obtain NDA access for HCP-Aging and NITRC DUA for OASIS-3; download a 300-participant age-stratified HCP-Aging subset (rfMRI + ASL + physio).
- [ ] Run rapidtide and rsHRF on the subset; compute parcel features; test-retest ICC (RQ1).
- [ ] Process pCASL to ATT; lag-vs-ATT validation with spin tests (RQ2).
- [ ] Full HCP-Aging run; normative model and hemodynamic-age model (RQ3).
- [ ] HCP-Aging vascular-risk associations (RQ4a); pre-register OASIS-3 analysis.
- [ ] fMRIPrep OASIS-3 sessions with rs-fMRI + FLAIR + PET; WMH segmentation; transfer and RQ4b.
- [ ] FC-age decomposition and mediation (RQ5); TR-downsampling simulation (RQ6).
- [ ] Manuscript 1 (reliability + ASL validation + normative chart) and manuscript 2 (vascular risk / WMH / amyloid / FC confound).

## Ethics / data-use notes

- HCP-Aging is distributed by the NIMH Data Archive under a Data Use Certification; data may not be redistributed and participant-level data must stay on approved systems. OASIS-3 requires the NITRC DUA and citation of the OASIS-3 acknowledgement text.
- No participant-level data, derivatives or credentials are committed; `data/` and `outputs/` are git-ignored. Credentials are read from environment variables only.
- The analysis needs no third-party LLM services; all processing is local.
- Report group-level statistics; do not publish individual lag maps that could be re-identified when combined with other released measures.
