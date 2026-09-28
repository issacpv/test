# Ideas Backlog — original project seeds

> **Status (2026-09-28): all 83 seeds below have been promoted to full project scaffolds under `projects/`.** Seed → folder mapping:
> 1 scanner-upgrade-discontinuity · 2 motion-causal-decomposition · 3 dmri-microstructure-charts · 4 cross-atlas-prediction-stability · 5 wmh-amyloid-interaction · 6 tau-proxy-from-mri · 7 sex-stratified-brain-age · 8 mriqc-scannability-equity · 9 openneuro-pet-kinetic-multiverse · 10 mri-foundation-model-clinical-validation · 11 fmri-hemodynamic-aging · 12 lesion-network-mapping-nulls · 13 cvr-normative-maps · 14 layer-fmri-7t-reproducibility · 15 radiology-report-weak-supervision-audit · 16 seizure-forecasting-eeg-ecg · 17 eeg-age-domain-shift · 18 interictal-spike-detection-benchmark · 19 anesthesia-depth-cross-dataset · 20 neonatal-eeg-maturation-clock · 21 sleep-staging-in-disease · 22 cross-species-spindle-coupling · 23 spike-sorter-multiverse · 24 laminar-latent-dynamics · 25 cross-modality-tuning-reproducibility · 26 human-concept-cell-reproducibility · 27 bci-decoder-transfer-benchmark · 28 meg-microstate-heritability · 29 neuromorpho-software-fingerprinting · 30 dendritic-spine-metadata-mining · 31 disease-morphology-signatures · 32 ais-plasticity-real-morphologies · 33 interneuron-morphology-ttype-transfer · 34 virtual-mouse-brain-validation · 35 celltype-composition-mri-contrast · 36 cross-species-imaging-transcriptomics · 37 gene-gradients-neural-timescales · 38 morphology-to-spike-waveform · 39 sepsis-definition-multiverse · 40 circadian-label-bias-icu · 41 lab-ordering-information-leak · 42 deid-residual-leakage-audit · 43 pulse-contour-cardiac-output · 44 arterial-line-damping-detection · 45 ventilator-asynchrony-detection · 46 qt-prolongation-dose-response · 47 ecg-foundation-model-probing · 48 pediatric-ecg-generalization · 49 echo-view-vendor-robustness · 50 cgm-forecasting-regimen-shift · 51 cuffless-bp-pregnancy · 52 perioperative-outcome-transportability · 53 ed-language-disparity · 54 radiology-delay-ed-disposition · 55 nursing-assessment-prediction · 56 bayesian-pk-real-world-dosing · 57 dynamic-delirium-risk · 58 fluid-responsiveness-waveforms · 59 ai-device-failure-taxonomy · 60 recall-lag-survival · 61 faers-ddi-signals · 62 biosimilar-ae-profiles · 63 glp1-sex-stratified-pv · 64 pediatric-offlabel-signals · 65 caers-supplement-signals · 66 drug-shortage-ae-patterns · 67 label-change-prediction · 68 animal-drug-cross-species-signals · 69 dbs-vta-population-variability · 70 retinal-prosthesis-morphology-models · 71 tes-dose-individualization · 72 cardiac-digital-twin-ecg-echo · 73 respiratory-mechanics-estimation · 74 oai-progression-uncertainty · 75 gait-wearable-cross-device · 76 neurofeedback-multiverse · 77 digital-health-rct-ipd-reanalysis · 78 leakage-detector-biomedical-ml · 79 unified-biomedical-shift-benchmark · 80 spatial-null-replication-audit · 81 effect-size-inflation-openneuro · 82 synthetic-ehr-transportability · 83 llm-cohort-extraction-reproducibility
>
> The project READMEs contain the *sharpened* angle after literature checks; the seeds below are kept as the original starting points.

Each is a one-paragraph seed with datasets and the gap. Promote to `projects/<slug>/` when you want to start it (copy the structure of any existing project). Ordered loosely by domain. "Gap" statements are working hypotheses — re-check the literature before committing.

## Neuroimaging

1. **Scanner-upgrade discontinuities in longitudinal cohorts** — OASIS-3 (Siemens Trio→Prisma / Vision), ADNI (1.5T→3T). Estimate the *counterfactual* atrophy trajectory across an upgrade with regression-discontinuity-in-time designs and compare harmonization methods (ComBat-GAM, longitudinal ComBat, DeepHarmony). Gap: harmonization papers are evaluated on traveling-subject sets, rarely on real cohort upgrades with clinical outcomes.
2. **Motion as a treatment: causal effect of head motion on FC-behavior associations** — HCP, ABCD, ABIDE. Use instrumental-variable / negative-control-outcome designs to bound how much of the "FC predicts cognition" effect is motion-mediated. Gap: motion is always "controlled for", never causally decomposed.
3. **Diffusion-MRI microstructure normative charts** — HCP (multi-shell), HCP-Aging/-Dev, Cam-CAN, IXI, OpenNeuro dMRI. NODDI/DKI lifespan curves are far rarer than volume charts. (Partially covered in `lifespan-normative-models`; this seed is the microstructure-only, model-comparison version.)
4. **Cross-atlas prediction stability** — HCP + OpenNeuro rest-fMRI: how much do CPM/ridge predictions and edge-level findings depend on parcellation (Schaefer 100–1000, Glasser, Gordon, AAL, Brainnetome)? Produce an atlas-robustness index per phenotype.
5. **White-matter hyperintensity × amyloid interaction on cognition** — OASIS-3 FLAIR + PiB + CDR. Mixed-effects with WMH-volume trajectories. Gap: OASIS-3 FLAIR is under-used vs ADNI.
6. **PET-free tau proxies** — OASIS-3 has AV1451 (tau) PET for a subset; predict tau positivity from T1 + FLAIR + plasma-free covariates. Extension of `amyloid-from-mri`.
7. **Sex-stratified brain-age** — HCP/OASIS/IXI/Cam-CAN: are brain-age models with sex-specific training more clinically informative? Related to reported sex bias in brain-age deltas.
8. **MRIQC-derived "scannability" and equity** — OpenNeuro participants.tsv + MRIQC IQMs: which populations (age, clinical groups) are systematically excluded by QC? (Extension of `openneuro-mriqc-audit`.)
9. **OpenNeuro PET reproducibility** — OpenNeuro's BIDS-PET datasets (kinetic modeling multiverse: SRTM vs Logan vs reference regions).
10. **Benchmarking foundation models for MRI (BrainSegFounder, SAM-Med3D, etc.) on OASIS/OpenNeuro clinical scans** — a strict external-validation study on pathology-rich, low-quality clinical data.
11. **fMRI hemodynamic aging** — HCP-Aging + OASIS-3 rest fMRI: HRF shape / vascular lag maps as aging & vascular-risk biomarkers (rarely modeled).
12. **Lesion-network mapping with rigorous nulls** — using OpenNeuro stroke datasets (e.g., ATLAS R2.0 on INDI) + HCP normative connectome; extends `tms-target-connectivity` to lesions.
13. **Cerebrovascular reactivity from BOLD in OpenNeuro breath-hold / CO2 datasets** — normative CVR maps, aging effects.
14. **7T vs 3T layer-fMRI reproducibility** — OpenNeuro 7T datasets; quantify laminar-profile reproducibility across labs.
15. **Radiology-report → imaging weak supervision on MIMIC-CXR / OpenNeuro clinical MRI** — silent label noise audit.

## Electrophysiology

16. **Seizure forecasting with wearables + EEG** — Siena (ECG available), plus PhysioNet "Epilepsy ECG" sets, and Empatica open datasets (e.g., "Seizure detection using wrist-worn sensors" on PhysioNet). Multimodal pre-ictal HRV. Gap: scalp-EEG + ECG joint models on open data.
17. **Pediatric→adult EEG domain shift as a function of age** — CHB-MIT (pediatric) vs TUSZ (all ages): performance as a smooth function of age difference; foundation-model embedding drift.
18. **Interictal spike detectors generalization** — TUEV, OpenNeuro iEEG, Bonn: automated spike detection benchmark.
19. **EEG-based anesthesia depth across datasets** — VitalDB (BIS + EEG), OpenNeuro anesthesia EEG: cross-dataset depth-of-anesthesia models and propofol vs sevoflurane transfer.
20. **Neonatal EEG maturation clocks** — Helsinki neonatal set + OpenNeuro neonatal EEG + NICU sets: "EEG age" vs postmenstrual age; deviation as outcome predictor.
21. **Sleep staging in disease** — NSRR (SHHS/MESA with AHI, CVD outcomes) vs Sleep-EDF: staging-model performance drop in OSA/CHF; N3 misclassification and its effect on outcome associations.
22. **Cross-species spindle/SO comparisons** — DANDI rodent sleep recordings (e.g., hippocampal/cortical LFP) vs human PSG: harmonized coupling metrics.
23. **Neuropixels spike-sorter multiverse** — DANDI raw AP-band recordings: Kilosort 2.5/3/4 vs SpikeInterface ensemble; effect on downstream science claims (drift, tuning).
24. **Cortical-layer specific latent dynamics** — Allen Visual Coding Neuropixels (layer-resolved units via CSD): do laminar populations occupy distinct latent subspaces?
25. **Cross-lab reproducibility of visual tuning** — Allen Visual Coding (Neuropixels vs 2-photon of the same stimuli): modality-dependent tuning estimates.
26. **Human single-neuron datasets on DANDI (e.g., Rutishauser lab memory tasks)** — concept-cell reproducibility across datasets.
27. **BCI decoder calibration transfer** — MOABB + PhysioNet EEGMMIDB: Riemannian alignment vs deep transfer across 30 datasets, effect sizes with mixed models.
28. **HRV & EEG microstates in HCP MEG** — resting MEG microstate dynamics vs heritability (twin design).

## Cellular / morphology / atlases

29. **NeuroMorpho reconstruction-software fingerprinting** — can a classifier identify the reconstruction software/lab from SWC geometry alone? Quantifies batch effects for meta-analyses.
30. **Dendritic spine density meta-data mining** — NeuroMorpho + literature; spines rarely reconstructed; audit.
31. **Aging and disease morphologies** — NeuroMorpho has AD-model, aging, epilepsy reconstructions: morphological signatures with lab-effect correction.
32. **Axon-initial-segment plasticity models fed by real morphologies** — NeuroMorpho + ModelDB.
33. **Interneuron morphology → transcriptomic-type prediction across species** — Allen Patch-seq mouse vs human MTG; how well do morphology-only embeddings recover t-types?
34. **Mesoscale connectome → whole-brain model validation** — Allen Mouse Connectivity + Allen fMRI/optical imaging (e.g., DANDI widefield): fit The Virtual Mouse Brain models and evaluate against empirical FC. Extends `mesoscale-vs-single-axon-connectivity`.
35. **Cell-type composition explains regional MRI signals** — ABC Atlas MERFISH densities vs mouse MRI atlases (e.g., AMBMC, DSURQE): what cell types drive T1/T2/diffusion contrast?
36. **Spatial transcriptomics × imaging transcriptomics cross-species null models** — ABC atlas + AHBA; test whether AHBA-based human associations replicate in mouse MERFISH-derived maps.
37. **Gene expression gradients vs Neuropixels tuning** — Allen ISH + Visual Coding: which genes predict regional timescales/tuning.
38. **Morphology-based estimation of extracellular potential signatures (spike waveform ↔ morphology)** — Allen Cell Types (morphology + ephys waveform) → predict waveform features from morphology; helps cell-type ID from Neuropixels.

## Clinical / ICU / signals

39. **Sepsis definition multiverse** — MIMIC-IV + eICU: Sepsis-3 implementation choices (SOFA baseline, infection window) change cohorts by >50%; quantify effect on model benchmarks.
40. **Time-of-day and staffing effects on ICU predictions** — MIMIC-IV timestamps: circadian label bias.
41. **Lab-ordering as an information leak** — MIMIC-IV: models learn from *which* labs were ordered; quantify and build order-agnostic models.
42. **MIMIC-IV-Note de-identification residual leakage audit** — local LLM/NER scan for PHI patterns (research-only; policy-compliant).
43. **Waveform-derived cardiac output surrogates** — MIMIC-IV Waveform ABP + echo (MIMIC-IV-Echo): pulse-contour analysis validated against echo-derived stroke volume.
44. **Arterial-line dampening detection** — MIMIC-III/IV waveform; signal-quality index for ABP; effect of damped waveforms on BP-based decisions.
45. **Ventilator waveform asynchrony detection** — VitalDB & MIMIC waveforms lack flow, but eICU/HiRID have high-frequency ventilator settings; alternatively Zenodo asynchrony datasets.
46. **Drug-induced QT prolongation dose-response from MIMIC-IV-ECG + emar** — fine-grained pharmaco-ECG.
47. **ECG foundation model probing** — PTB-XL/MIMIC-IV-ECG: what do embeddings encode (age, sex, race proxies) and fairness implications.
48. **Pediatric ECG generalization** — CODE-15 includes children; PTB-XL has few; transfer to pediatric.
49. **Echo view classification robustness** — TMED-2 / EchoNet: vendor shift.
50. **Continuous glucose datasets (OhioT1DM, Tidepool, OpenAPS Data Commons)** — glucose forecasting under regimen shift; DIY-loop data audits.
51. **Cuffless BP in pregnancy / hypertensive disorders** — VitalDB obstetric cases (if any) + MIMIC-IV obstetrics: domain where PTT models are least validated.
52. **Perioperative outcome prediction transportability** — VitalDB + INSPIRE (PhysioNet) + MIMIC-IV surgical cohort.
53. **ED chief-complaint language & interpreter use** — MIMIC-IV-ED language field: outcomes disparity.
54. **Radiology report timing vs ED disposition** — MIMIC-IV-ED + MIMIC-CXR timestamps: does report delay predict outcome?
55. **Hospital-acquired pressure-injury / fall prediction** — MIMIC-IV chartevents nursing assessments (Braden/Morse scales) — under-used.
56. **Bayesian PK/PD from MIMIC-IV emar + labs** — vancomycin/aminoglycoside levels: population PK models with real-world dosing.
57. **Sedation and delirium** — MIMIC-IV CAM-ICU + RASS: dynamic delirium risk.
58. **Fluid-responsiveness from waveforms** — MIMIC-IV Waveform + inputevents boluses: PPV/SVV predictors of bolus response.

## Regulatory / pharmacovigilance

59. **openFDA device: SaMD/AI-device MAUDE narratives** — NLP taxonomy of AI-device failure modes; cross-link with FDA AI-device list (partially in `device-recall-prediction`).
60. **Recall-to-MAUDE lag time modeling** — survival analysis of time between first serious MAUDE event and recall by device class/company.
61. **FAERS reporting of drug–drug interactions** — TWOSIDES-style signal detection with modern methods (shrinkage, Bayesian hierarchical), sex-stratified.
62. **Biosimilar vs originator adverse-event profiles** — FAERS + label data.
63. **GLP-1 agonists pharmacovigilance across sexes** — timely, FAERS + labels.
64. **Pediatric off-label use signals** — FAERS age fields + label pediatric sections.
65. **openFDA food/dietary supplements adverse events (CAERS)** — very under-studied.
66. **Drug shortage (FDA/ASHP lists) → adverse-event patterns** — substitution errors visible in FAERS?
67. **Label-change prediction** — SPL version history: predict which labels get a boxed-warning update from FAERS trajectories (extension of `unlabeled-adverse-event-mining`).
68. **Animal drug adverse events (openFDA animalandveterinary)** — cross-species toxicity signals vs human FAERS.

## BME modeling / devices

69. **Population-level DBS VTA variability** — NeuroMorpho + open DBS lead models (Lead-DBS) + Allen human morphologies.
70. **Cochlear-implant or retinal-prosthesis population models with real morphologies** — NeuroMorpho retinal ganglion cells.
71. **tES dose individualization from MRI** — OpenNeuro tDCS datasets with T1: SimNIBS E-field vs behavioral effect, spatial nulls.
72. **Cardiac digital twins from EchoNet + PTB-XL** — joint ECG–echo generative models; validate on MIMIC-IV-ECG + MIMIC-IV-Echo.
73. **Respiratory mechanics estimation from ventilator time series** — HiRID / eICU respiratory charting: compliance/resistance inference.
74. **Bone/joint: OAI (Osteoarthritis Initiative) MRI/X-ray progression** — open via NDA-lite; deep learning + uncertainty.
75. **Gait/wearables: PhysioNet gait datasets + Parkinson's (PPMI wearables)** — cross-device generalization.
76. **EEG-based neurofeedback efficacy reanalysis** — OpenNeuro neurofeedback datasets: multiverse of feedback-signal computation.
77. **Digital-health RCT re-analysis with individual participant data** — Vivli/YODA (application).

## Cross-cutting methods projects (publishable in methods venues)

78. **Leakage detector for biomedical ML repos** — static analysis of published GitHub code for subject-level leakage patterns; evaluate on papers using CHB-MIT/PTB-XL/MIMIC.
79. **A unified "shift benchmark" across modalities** — combine `icu-model-transportability`, `ecg-cross-dataset-generalization`, `cross-dataset-seizure-generalization`, `echonet-ef-uncertainty` into one WILDS-style biomedical shift benchmark.
80. **Spatial-null model software audit** — extension of `imaging-transcriptomics-nulls`: replicate 50 published brain-map correlations with corrected nulls.
81. **Effect-size inflation in small-n neuroimaging on OpenNeuro** — winner's-curse estimation using OpenNeuro's dataset sizes vs claimed effects.
82. **Synthetic-data fidelity for EHR (Synthea, MIMIC-derived GANs)** — downstream-model transportability from synthetic to real.
83. **LLM extraction of cohort definitions from papers → reproducibility on MIMIC** — with local models only, per PhysioNet policy.
