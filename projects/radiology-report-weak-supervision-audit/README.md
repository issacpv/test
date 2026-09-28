# radiology-report-weak-supervision-audit — Silent, differential label noise in report-derived chest X-ray labels and what it does to benchmarks and fairness claims

Quantify, across every open expert image-level label set that overlaps or parallels MIMIC-CXR, how far report-derived labels (CheXpert labeler, NegBio, CheXbert, local LLM labelers) sit from image truth for each finding; test whether that disagreement is differential by view, care setting, sex, race/ethnicity, age and insurance; and measure, by calibrated simulation and re-evaluation of public models, how much of published benchmark rankings and "underdiagnosis bias" gaps can be produced by the label noise alone.

## Status / difficulty / timeline / compute

- Status: design + starter code (transparent rule-based report labeler with report-structure features, agreement and differential-noise statistics, noise-rate estimation and fairness/AUC simulation, a local-only LLM labeler wrapper).
- Difficulty: MSc-to-PhD, 6-9 months. Mostly tabular/NLP; image models are only re-evaluated, not trained.
- Compute: CPU for everything except (a) CheXbert inference (minutes on one GPU) and (b) the optional local LLM labeler (one 24-80 GB GPU for a 7-70B model over ~230k reports; days). Storage: MIMIC-CXR-JPG ~570 GB (can be limited to the expert-labeled studies, < 20 GB).

## Background

Chest X-ray AI is trained and, worse, evaluated on labels extracted automatically from radiology reports: MIMIC-CXR (Johnson et al., 2019, Scientific Data) ships CheXpert-labeler (Irvin et al., 2019, AAAI) and NegBio (Peng et al., 2018, AMIA Summits) labels; CheXbert (Smit et al., 2020, EMNLP) improved extraction; VisualCheXbert (Jain et al., 2021, CHIL) showed explicitly that report labels differ from image-level labels and trained a mapping to the latter; RadGraph (Jain et al., 2021, NeurIPS Datasets and Benchmarks) structured the reports. Label errors in public datasets were flagged early (Oakden-Rayner, 2020, Academic Radiology) and, generically, test-set label errors are known to destabilize benchmarks (Northcutt, Athalye and Mueller, 2021, NeurIPS Datasets and Benchmarks; confident learning, Northcutt, Jiang and Chuang, 2021, JAIR). On fairness, under-served groups showed higher false-negative "no finding" rates (Seyyed-Kalantari et al., 2021, Nature Medicine), a result whose interpretation was contested because dataset biases, including label noise, were not separated from model behaviour (Bernhardt, Jones and Glocker, 2022, Nature Medicine); models also encode protected characteristics (Glocker et al., 2023, eBioMedicine). Privacy-preserving LLM labelers running locally have been shown feasible on MIMIC reports (Mukherjee et al., 2023, Radiology, Vicuna), and PhysioNet forbids sending credentialed notes to external LLM APIs outside its responsible-use guidance.

## The research gap

What has been done (2024-2026):

- "Limitations of Public Chest Radiography Datasets for Artificial Intelligence: Label Quality, Domain Shift, Bias and Evaluation Challenges" (arXiv 2509.15107, 2025): in a 120-case MIMIC-CXR re-read, two experts disagreed with the dataset labels in 58 and 49 cases, mostly in the experts' favour.
- "When Repository Labels Are Not Image-Level Truth: A Supervision Auditing Framework for Chest Radiograph AI" (arXiv 2608.10084, 2026): cardiomegaly in MIMIC-CXR as a case study; near-zero agreement between repository and expert image-level labels, driven by non-mention rather than explicit negation.
- Labeler-vs-labeler comparisons on reports (e.g. JMIR Medical Informatics, 2025, "Automated radiology report labeling in chest X-ray...") and LLM-based labeling papers, which measure extraction accuracy against report annotations, not against images.
- Long-tail (CXR-LT) and multi-dataset evaluations that inherit the same weak labels.

What is specifically missing:

1. A multi-finding, multi-source noise audit at scale. Expert image-level labels already exist for thousands of MIMIC-CXR images (Chest ImaGenome gold standard, REFLACX, MS-CXR) and for parallel datasets (CheXpert validation/test, VinDr-CXR with three radiologists, PadChest's manually labelled 27%, the NIH adjudicated subset); nobody has assembled them into per-finding, per-labeler noise (confusion) matrices with confidence intervals.
2. Differential noise. Whether report-to-image disagreement depends on view/portable acquisition, care setting (ICU vs ED vs ward), sex, race/ethnicity, age, insurance and language has never been measured, although MIMIC-CXR links to MIMIC-IV for all of these. Non-mention false negatives plausibly concentrate in ICU portable films written as "unchanged" comparisons, which are unevenly distributed across demographic groups.
3. Consequences. No study has (a) simulated, with noise rates calibrated to the audit, how much subgroup FNR gap a perfect classifier would display, or (b) re-ranked public CXR models on expert vs report labels with rank-correlation statistics.
4. Report-structure drivers (templated normals, comparison language, hedging) as predictors of silent noise, and whether local LLM labelers change the *differential* structure of the noise rather than its overall level.
5. The MRI analogue is not currently testable on OpenNeuro (clinical MRI datasets there rarely ship radiology reports); this project therefore focuses on CXR and lists CT/MRI report-paired datasets as follow-ups.

## Research questions / hypotheses

1. RQ1 (noise matrices). For the 14 CheXpert findings and each labeler (CheXpert, NegBio, CheXbert, `cxr_label_audit` rules, local LLM), what are sensitivity, specificity and κ against expert image labels, treating non-mention as negative (the standard training convention) vs as missing? H1: sensitivity < 0.6 for cardiomegaly, atelectasis and pleural effusion under the non-mention-as-negative convention; specificity > 0.9; uncertainty labels ("-1") are informative (positive rate 40-70%).
2. RQ2 (differential noise). Disagreement (image-positive, report-negative/unmentioned) is higher for portable AP films and ICU studies; after adjusting for view and care setting, residual associations with sex, race/ethnicity, insurance and age are small. H2: OR(ICU portable vs PA outpatient) > 2 for non-mention FNs; adjusted demographic ORs within 0.8-1.25.
3. RQ3 (report drivers). Comparison language ("unchanged", "stable", "again") and templated normal impressions carry most non-mention FNs. H3: OR > 3 for comparison-style reports.
4. RQ4 (fairness impact). Simulations with the measured, subgroup-specific noise rates applied to a perfect classifier reproduce 30-60% of the subgroup "no finding" FNR gaps reported for MIMIC-CXR.
5. RQ5 (benchmark impact). Re-evaluating ≥ 5 public CXR models (e.g. TorchXRayVision weights, excluding models trained on MIMIC-CXR for MIMIC test images) on expert vs report labels yields Kendall τ < 0.7 in model rankings for at least three findings.
6. RQ6 (LLM labelers). A local LLM labeler reduces negation/uncertainty errors but not non-mention FNs, so the differential structure (RQ2) persists.

## Datasets

| Dataset | What is used | Size (approx.) | Access | URL |
|---|---|---|---|---|
| MIMIC-CXR v2.1 / MIMIC-CXR-JPG v2.1 (Johnson et al., 2019, Scientific Data) | Free-text reports, DICOM/JPG images, CheXpert and NegBio label tables, metadata (view, procedure) | 377k images, 228k studies, 65k patients | PhysioNet credentialed (CITI training + DUA) | https://physionet.org/content/mimic-cxr-jpg/ |
| MIMIC-IV v3 (Johnson et al., 2023, Scientific Data) | Care setting at study time (transfers/ICU stays, ED), sex, age, race/ethnicity, insurance, language | 300k+ patients | PhysioNet credentialed | https://physionet.org/content/mimiciv/ |
| Chest ImaGenome v1.0.0 (Wu et al., 2021, NeurIPS Datasets and Benchmarks) | Scene graphs on MIMIC-CXR; gold-standard manually validated subset (500 studies) | 242k images; 500 gold studies | PhysioNet credentialed | https://physionet.org/content/chest-imagenome/ |
| REFLACX v1.0.0 (Bigolin Lanfredi et al., 2022, Scientific Data) | Radiologist image-level labels + eye tracking on MIMIC-CXR images | ~3,000 images | PhysioNet credentialed | https://physionet.org/content/reflacx-xray-localization/ |
| MS-CXR (Boecking et al., 2022, ECCV) | Radiologist bounding boxes + phrases on MIMIC-CXR | 1,162 annotations | PhysioNet credentialed | https://physionet.org/content/ms-cxr/ |
| VinDr-CXR (Nguyen et al., 2022, Scientific Data) | 3-radiologist image labels (no reports); inter-rater reference | 18,000 images | PhysioNet credentialed | https://physionet.org/content/vindr-cxr/ |
| CheXpert (Irvin et al., 2019, AAAI) | Report-derived labels + radiologist-labelled validation (200 studies) / test (500 studies) | 224k images | Free registration (Stanford AIMI) | https://stanfordaimi.azurewebsites.net/ |
| PadChest (Bustos et al., 2020, Medical Image Analysis) | 27% physician-labelled reports vs NLP-labelled remainder (Spanish) | 160k images | Free registration | https://bimcv.cipf.es/bimcv-projects/padchest/ |
| NIH ChestX-ray14 + adjudicated labels (Wang et al., 2017, CVPR; Majkowska et al., 2020, Radiology) | NLP labels vs radiologist-adjudicated subset | 112k images; ~4,000 adjudicated | Open | https://nihcc.app.box.com/v/ChestXray-NIHCC |

## Methods

1. Ontology harmonisation. Map every expert label set to the 14 CheXpert findings (with an explicit mapping table and "not assessable" flags); keep finding-specific caveats (e.g. cardiomegaly on AP films).
2. Report labelling (`cxr_label_audit.report_labeler` for a transparent, dependency-free baseline; production labelers: CheXpert labeler, NegBio, CheXbert). Section split (FINDINGS/IMPRESSION), sentence-level mention detection, negation and uncertainty scoping; report-structure features: word count, comparison language, templated normal impression, hedging density, limited-study phrases.
3. Local LLM labeler (`cxr_label_audit.local_llm_labeler`): JSON-schema prompt, strict loopback/private-network URL guard (vLLM/Ollama on-prem only), deterministic decoding, parse and audit trail. No report ever leaves the secure environment.
4. Metadata joins: MIMIC-CXR metadata (ViewPosition, PerformedProcedureStepDescription → portable), study time joined to MIMIC-IV `transfers`/`icustays`/`edstays` for care setting, `patients`/`admissions` for sex, age, race, insurance, language.
5. Agreement (`cxr_label_audit.agreement`): sensitivity/specificity/PPV/κ/PABAK per finding and labeler, under both non-mention conventions; patient-clustered bootstrap CIs; differential-noise logistic models (disagreement ~ view + portable + setting + sex + race + age + insurance + report features) with cluster-robust SEs; Holm across findings.
6. Simulation (`cxr_label_audit.noise_simulation`): estimate noise rates (from the expert subsets, and via confident learning from model probabilities on the full set); inject subgroup-specific noise into true labels; compute FNR gaps and AUC of a perfect and of realistic classifiers; compare to published gaps.
7. Benchmark re-ranking: run public models locally on the expert-labelled images (excluding models trained on those datasets); AUC per finding on expert vs report labels; Kendall τ and bootstrap CIs for rank changes.

Libraries: pandas, numpy, scipy, statsmodels, scikit-learn; regex; optional torch + torchxrayvision, CheXbert, negbio, chexpert-labeler; vLLM or Ollama for the local LLM.

## Evaluation and statistics

- Agreement metrics with 2,000-resample patient-clustered bootstrap CIs; κ and PABAK reported together because prevalence is low.
- Differential noise: logistic regression with cluster-robust SEs (patient), adjusted ORs with Holm correction over the 14 findings × 6 covariates family; E-values for unmeasured confounding.
- Simulations: 1,000 replications per condition; report the fraction of the published gap reproduced with 95% intervals.
- Benchmark: DeLong/bootstrap AUC CIs; Kendall τ of rankings across findings.
- Leakage prevention: every public model's training data is checked against the evaluation images (study-level); expert subsets are never used to tune labelers; the local LLM prompt is frozen before evaluation.
- Nulls: permutation of subgroup labels within care setting for the differential-noise test; permutation of expert labels for κ.
- Sensitivity analyses: alternative ontology mappings; excluding AP films for cardiomegaly; restricting to studies with ≥ 2 radiologists (VinDr, CheXpert).

## Publishable angle

Headline: "Report-derived chest X-ray labels are not missing at random: non-mention false negatives concentrate in ICU portable studies and comparison-style reports, are distributed unevenly across demographic groups, reproduce a substantial share of reported underdiagnosis gaps in simulation, and reorder public model rankings; local LLM labelers do not fix this because the information is absent from the report, not mis-extracted." Deliverables: per-finding noise matrices, a differential-noise table, a noise-injection simulator and a reporting checklist for weak-label benchmarks.

Target venues: Radiology: Artificial Intelligence; npj Digital Medicine; JAMIA; Medical Image Analysis; CHIL / ML4H / MIDL for an early version.

Follow-ups: noise-aware training (loss correction with the estimated matrices; co-teaching) and re-evaluation of fairness after correction; extension to report-paired CT/MRI datasets (e.g. chest CT with reports such as CT-RATE, subject to their licences) since OpenNeuro clinical MRI rarely includes reports; a PhysioNet-compliant benchmark of local LLM labelers.

## Risks, confounds and mitigations

- Expert labels are themselves noisy (VinDr and CheXpert show substantial inter-rater disagreement): use multi-rater consensus where available, report inter-rater κ alongside, and treat single-rater subsets (REFLACX) as lower-tier evidence.
- Expert subsets are not random samples of MIMIC-CXR (enrichment for abnormalities, front views): reweight by view/setting strata and state the target population.
- Care setting and demographics are confounded: the analysis models setting explicitly and reports both crude and adjusted ORs; causal claims are avoided.
- Small counts for rare findings (fracture, pneumothorax): pool or drop with pre-specified rules.
- Race categories in MIMIC-IV are coarse and partly missing: analyse missingness itself as a variable.
- PhysioNet policy: notes and images stay inside the credentialed environment; the LLM wrapper refuses non-local endpoints; no data or outputs containing text are committed.
- Public model leakage (TorchXRayVision MIMIC weights): excluded from MIMIC evaluations; documented per model.

## Milestones

- [ ] Obtain PhysioNet credentials and DUAs (MIMIC-CXR-JPG, MIMIC-IV, Chest ImaGenome, REFLACX, MS-CXR, VinDr-CXR); register for CheXpert and PadChest; download the NIH adjudicated labels.
- [ ] Ontology mapping table; assemble the expert-label master table with patient/study keys.
- [ ] Run rule-based, CheXpert, NegBio, CheXbert labelers on all MIMIC-CXR reports; store report-structure features.
- [ ] Noise matrices and κ per finding/labeler (RQ1); pre-register RQ2-RQ5.
- [ ] MIMIC-IV joins; differential-noise models (RQ2, RQ3).
- [ ] Noise-injection simulation of fairness metrics (RQ4); benchmark re-ranking (RQ5).
- [ ] Optional local LLM labeler run (RQ6); manuscript and released tables/simulator.

## Ethics / data-use notes

- MIMIC-CXR, MIMIC-IV, Chest ImaGenome, REFLACX, MS-CXR and VinDr-CXR are PhysioNet credentialed resources: CITI training, signed DUAs, storage on approved systems, no redistribution, and no sharing of notes or images with third-party services. Radiology reports must not be sent to external LLM APIs; only local models (or services explicitly permitted by PhysioNet's responsible-use guidance) may process them, which the code enforces.
- No data, labels or report text are committed; `data/` and `outputs/` are ignored. Released artefacts are aggregate tables and code.
- Demographic analyses follow the MIMIC-IV variable definitions and are reported with their limitations; no attempt is made to infer protected attributes.
