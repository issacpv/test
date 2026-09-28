# ECG-Probe: what do ECG foundation-model embeddings encode about demographics, and does it matter for fairness?

**One-sentence pitch.** Systematically probe the frozen embeddings of open 12-lead ECG foundation models (ECG-FM, HuBERT-ECG, ECGFounder) on PTB-XL and MIMIC-IV-ECG for how strongly they encode age, sex and self-reported race/ethnicity, locate *where* in the network that information lives (layer-wise probing), test whether removing it with concept-erasure changes diagnostic accuracy and subgroup fairness, and connect demographic leakage to concrete equity harms in downstream diagnosis.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data or model weights are shipped.
- Difficulty: MSc-to-early-PhD (representation learning + fairness + biostatistics).
- Timeline: 6-9 months (1-2 months embedding extraction + probing harness, 2 months layer-wise probes + concept erasure, 2 months fairness/downstream link, 1 month MIMIC race-proxy analysis, remainder writing).
- Compute: one 16-24 GB GPU to extract frozen embeddings for ~150k ECGs (hours); all probing/erasure/statistics run on CPU. Fine-tuning is optional. Disk: PTB-XL ~3 GB; MIMIC-IV-ECG ~90 GB (or use the machine-measurement subset + a waveform sample).

## Background

Deep ECG models can infer patient age (MAE ~7 y) and sex (AUROC >0.9) from the waveform alone (Attia et al., 2019, Circulation; Lima et al., 2021, Nat Commun), and imaging models can recover self-reported race from signals that clinicians cannot use (Gichoya et al., 2022, Lancet Digital Health, for radiology). If a foundation model's *general-purpose* embedding silently encodes protected attributes, every downstream classifier built on it inherits a shortcut: the representation can route diagnostic decisions through demographic proxies, producing subgroup performance gaps and privacy risk even when the downstream label has no legitimate demographic dependence.

Open ECG foundation models now exist - ECG-FM (McKeen et al., 2024/2025, *JAMIA Open*, doi:10.1093/jamiaopen/ooaf122), HuBERT-ECG (Coppola et al., 2024, medRxiv), ECGFounder (Li et al., 2024, arXiv:2410.04133) - and their own papers report that embeddings encode demographics (ECG-FM: gender AUC ~0.97, age MAE ~10.4 y) and that training on single-sex or single-race cohorts still yields ">=0.90 AUROC" across groups. But those reports are (a) at the final embedding only, (b) framed as reassuring "robustness", and (c) not tied to a *layer-resolved* account of where demographic information enters, nor to an *interventional* test of whether erasing it helps or hurts.

## The research gap

**What has been done (2022-2026):**

- **Demographic prediction from ECG** is well established (Attia 2019; Lima 2021, Nat Commun; and echo: Deep Learning Discovery of Demographic Biomarkers in Echocardiography, arXiv:2207.06421).
- **Race disparities in ECG models**: "Race, Sex and Age Disparities in the Performance of ECG Deep Learning Models Predicting Heart Failure" (medRxiv 2023, 2023.05.19.23290257) documents subgroup gaps but does not probe embeddings or intervene.
- **Foundation-model self-reports**: ECG-FM (2025) and DeepECG-SSL (2025) report demographic-inference numbers and small fairness gaps (TPR/FPR differences <0.01 across age/sex) but at the output/embedding level only.
- **Privacy-preserving ECG**: a 2026 study (Sci Rep, s41598-026-47665-6) shields age/sex/race in ECG representations while preserving diagnostic signal, and "Non-genetic factors determine deep-learning-identified ECG..." (PMC12513828) argues race predictability is driven by non-genetic factors - both establish that leakage exists and can be reduced, but neither maps it layer-by-layer across *multiple public foundation models*, nor links the erasable component to downstream subgroup fairness on a common benchmark.
- **NLP concept-erasure methods** (INLP, Ravfogel et al., 2020; RLACE/LEACE, Belrose et al., 2023, NeurIPS) give principled linear-erasure tools that have not been applied to ECG foundation-model embeddings.

**What is specifically missing (the gap this project fills):**

1. A **layer-wise, multi-model probing map**: for each open foundation model, how much age/sex/race information is *linearly* and *non-linearly* decodable at each block, using minimum-description-length (MDL) probing (Voita & Titov, 2020) so probe capacity is controlled.
2. An **interventional** test: apply LEACE/INLP concept erasure to the frozen embedding, then measure (a) how much diagnostic AUROC is lost and (b) whether subgroup fairness gaps shrink - separating "demographics as harmful shortcut" from "demographics as legitimate physiological covariate."
3. A **downstream-harm link**: quantify, across the PhysioNet-2021 diagnostic label set, the correlation between a label's *demographic decodability from the embedding* and its *subgroup performance gap*, i.e. does more leakage predict more inequity?
4. A careful **race-proxy analysis in MIMIC-IV-ECG** (which has self-reported race) with the explicit caveat that race is a social, not biological, variable; report what is decodable and, per PhysioNet policy, keep all analysis local.

This is distinct from the sibling project `ecg-cross-dataset-generalization` (which studies cross-country transfer and subgroup *parity* of task performance); here the object of study is the *representation itself* and the interventional erasure.

## Research questions / hypotheses

1. **RQ1 (encoding strength).** How accurately are age (MAE, R^2), sex (AUROC) and self-reported race/ethnicity (macro-AUROC, in MIMIC) linearly decodable from each frozen foundation model's final embedding? *H1:* sex AUROC >0.95, age MAE <8 y, race macro-AUROC >0.7 for at least one model - well above chance - confirming substantial encoding.
2. **RQ2 (layer localisation).** At which depth does demographic information peak? *H2:* sex/age information rises monotonically to the final layers (task-aligned), whereas low-level acquisition-linked information (and part of the race signal) peaks earlier; MDL probes show the sex code is compressible (low description length) while race is diffuse.
3. **RQ3 (erasability vs utility).** After LEACE erasure of a target attribute, how much diagnostic AUROC is lost per unit of attribute-decodability removed? *H3:* sex can be linearly erased with <2% average diagnostic AUROC loss for most labels, but labels with genuine sex-dependence (e.g., LVH criteria) lose more - identifying where demographics are legitimate.
4. **RQ4 (leakage -> inequity).** Across diagnostic labels, does higher demographic decodability from the embedding predict a larger subgroup performance gap of a linear probe trained for that label? *H4:* positive Spearman correlation (rho>0.3) between race/age decodability and the age/race AUROC gap.
5. **RQ5 (model comparison).** Do self-supervised (HuBERT-ECG) vs supervised-pretrained (ECGFounder) vs hybrid (ECG-FM) embeddings differ in how much protected-attribute information they carry at matched diagnostic utility? *H5:* supervised-pretrained embeddings encode less non-task demographic information at matched AUROC.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| PTB-XL v1.0.3 | Waveforms + age, sex, SCP-ECG diagnostic labels | 21,799 ECGs, 18,869 patients, 500 Hz, 10 s | Open (PhysioNet) | https://physionet.org/content/ptb-xl/1.0.3/ |
| PTB-XL+ (features) | Optional 12SL/median-beat features for interpretability | same cohort | Open (PhysioNet) | https://physionet.org/content/ptb-xl-plus/1.0.1/ |
| MIMIC-IV-ECG v1.0 | Waveforms; self-reported race/ethnicity + age/sex via linkage | ~800k ECGs, ~160k patients | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| MIMIC-IV v3.1 hosp | `admissions.race`, `patients` (anchor_age, gender) | ~365k patients | Credentialed (PhysioNet) | https://physionet.org/content/mimiciv/3.1/ |
| PhysioNet/CinC 2021 training | Optional external diagnostic labels (SNOMED) for RQ4 | ~88k ECGs | Open | https://physionet.org/content/challenge-2021/1.0.3/ |

Model weights (verify licences before redistribution): ECG-FM (open), HuBERT-ECG (open), ECGFounder (open, research). A supervised 1D-ResNet trained from scratch on PTB-XL is the non-foundation baseline embedding.

## Methods

Pipeline (`src/ecg_probe/`):

1. **Embedding extraction** (`embeddings.py`). Harmonise each ECG to (12, 5000) at 500 Hz, standard lead order, per-record z-scoring; run each frozen model to obtain the final embedding and (where architecture allows) per-block hidden states. A model registry abstracts each backbone behind a common `encode(x) -> dict[layer, vector]` interface; a deterministic random-projection "mock encoder" is shipped so the harness and tests run without downloading weights.
2. **Probes** (`probing.py`). For each (model, layer, attribute): a linear probe (logistic/ridge) and a small MLP probe with nested CV; **MDL / online-code probing** (Voita & Titov 2020) to control probe capacity and report codelength in bits, plus a control task (random-label) to compute *selectivity* (Hewitt & Liang 2019). Patient-level splits throughout.
3. **Concept erasure** (`erasure.py`). LEACE (Belrose et al., 2023) closed-form linear erasure and iterative INLP as a comparator; produce an erased embedding, re-measure attribute decodability (should fall to chance) and re-train diagnostic probes to measure the utility cost.
4. **Fairness link** (`fairness.py`). Train per-label diagnostic linear probes on original vs erased embeddings; compute subgroup AUROC gaps (age band, sex, race) with stratified bootstrap CIs and equalised-odds gaps; correlate label-wise leakage with label-wise gap (RQ4).
5. **Statistics/nulls** (`stats.py`). Permutation null for probe accuracy (shuffle attribute labels), DeLong tests for AUROC differences, bootstrap CIs, Benjamini-Hochberg across labels/layers.

Tools: `numpy`, `scipy`, `scikit-learn`, `pandas`, `wfdb`, `statsmodels`; optional `torch` + `transformers`/model repos for real backbones. Concept erasure implemented from the closed-form LEACE equations (no heavy dependency required).

## Evaluation & statistics

- **Probe metrics.** Attribute decodability: AUROC (sex, race one-vs-rest), MAE and R^2 (age); MDL codelength (bits) and selectivity (probe minus control-task accuracy). Chance baselines from label permutation.
- **Erasure metrics.** Post-erasure attribute AUROC (target ~0.5), guardedness (linear-guardedness certificate from LEACE), and diagnostic-utility delta (AUROC before minus after) per label.
- **Fairness metrics.** Subgroup AUROC gap = max-min across groups; equalised-odds gap at a fixed operating point; calibration (ECE) per group. Stratified bootstrap (2000 resamples, patient-level).
- **Validation scheme.** Patient-level train/val/test; PTB-XL official folds (1-8 train, 9 val, 10 test); MIMIC split by subject_id. Probes never see test patients; erasure projections fit on train only and applied to test.
- **Leakage prevention.** No demographic variable is an input to any model; probes decode from embeddings only. Same-patient ECGs kept within one split. Race is analysed only where self-reported (MIMIC) and never inferred to label individuals.
- **Multiple comparisons.** models x layers x attributes x labels: Benjamini-Hochberg FDR within each RQ family; effect sizes with CIs primary.
- **Nulls.** (i) Permuted-attribute probe (chance decodability). (ii) Random-projection mock encoder (should show only trivially decodable age/sex from gross morphology, a floor). (iii) Control-task selectivity to rule out probe memorisation.

## Publishable angle

- **Headline.** "Across three open ECG foundation models, sex and age are highly and compressibly encoded throughout the network while self-reported race is diffusely decodable; linear concept-erasure removes sex/age with small, label-specific utility cost, and labels whose embeddings leak more demographics show larger subgroup gaps - a direct link from representation leakage to diagnostic inequity."
- **Deliverables.** A layer-wise leakage atlas per model, the erasure-vs-utility trade-off curves, the leakage-to-gap correlation, and an open probing/erasure toolkit for ECG embeddings.
- **Target venues.** *ML4H* / *CHIL* (proceedings); *npj Digital Medicine*; *IEEE JBHI*; *Journal of Biomedical Informatics*; fairness venues (*FAccT*) for the equity-link result.
- **Follow-ups.** Nonlinear/kernelised erasure; extend to wearable single-lead embeddings; causal mediation of the demographic pathway into specific diagnoses; auditing commercial ECG algorithms via their exported features.

## Risks, confounds & mitigations

- **"Encoding" != "harm".** Some demographic dependence is physiologically legitimate (age/sex genuinely change the ECG). Mitigation: the interventional erasure + per-label utility cost distinguishes legitimate covariance from shortcut; do not claim harm without the RQ4 link.
- **Probe capacity confound.** A powerful probe can decode noise. Mitigation: MDL/online-code probing + control-task selectivity.
- **Race is socially constructed and unevenly recorded** in MIMIC. Mitigation: treat as self-reported social category; report missingness; never use to label individuals; frame findings as bias/privacy audit.
- **Foundation-model pretraining overlap** with PTB-XL/MIMIC inflates decodability. Mitigation: document each model's pretraining corpus; treat overlapping data as in-distribution and flag it.
- **Erasure only guards against linear adversaries.** Mitigation: report the LEACE linear-guardedness caveat and add a nonlinear-probe check post-erasure.

## Milestones

- [ ] Download PTB-XL; build the harmonised loader and the mock-encoder harness; run tests.
- [ ] Integrate at least two real foundation-model backbones; extract frozen embeddings (PTB-XL).
- [ ] Linear + MLP + MDL probes for age/sex; layer-wise maps; selectivity controls.
- [ ] LEACE/INLP erasure; utility-cost curves on PTB-XL diagnostic labels.
- [ ] PhysioNet credentialing; MIMIC-IV-ECG embeddings; race-proxy probing (local only).
- [ ] Fairness link (RQ4) with bootstrap CIs; model comparison (RQ5).
- [ ] Toolkit release + manuscript.

## Ethics / data-use notes

- MIMIC-IV / MIMIC-IV-ECG are credentialed: CITI training + DUA; encrypt at rest; **never** send waveforms, embeddings derived from them, or race labels to third-party LLM/API services except per PhysioNet's responsible-use policy. Never commit data or per-record embeddings.
- Credentials read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD` env vars only.
- Race/ethnicity is analysed to expose and mitigate bias and privacy risk, not to build demographic classifiers for deployment; the released code disables the race-inference path by default and documents the harm.
- Foundation-model licences vary (some research-only); check before redistributing embeddings or fine-tuned weights.
- Related project in this repo: `ecg-cross-dataset-generalization` (population-shift task benchmark) - complementary; keep this project self-contained.
