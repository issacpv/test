# ECG-XGen: a population-shift leaderboard for 12-lead ECG classifiers

**One-sentence pitch.** Build the first *population-shift* benchmark for deep 12-lead ECG classifiers across seven public datasets from four countries (Germany, China, USA, Brazil), decompose the cross-dataset performance drop into label-mapping, device/sampling-rate and population components, evaluate open ECG foundation models on the same transfer grid with label-noise-aware and subgroup-parity metrics, and validate the surviving models against hard clinical outcomes (mortality, troponin elevation) in MIMIC-IV-ECG.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are shipped.
- Difficulty: MSc thesis to early-PhD level (signal processing + deep learning + biostatistics).
- Timeline: 6-9 months (2 months harmonisation + baselines, 2 months transfer grid + foundation models, 2 months MIMIC-IV-ECG outcome validation + statistics, remainder writing).
- Compute: one 24 GB GPU is sufficient for 1D-ResNet training on ~130k ECGs (hours per run). Fine-tuning ECG foundation models (ECG-FM ~300M params; HuBERT-ECG base) is feasible on a single A100/L40S; frozen-embedding evaluation runs on any GPU. MIMIC-IV-ECG (800k records, ~90 GB WFDB) needs ~150 GB disk. CPU-only is possible for the harmonisation, shift diagnostics and statistics.

## Background

Deep ECG classifiers reach cardiologist-level AUROC *within* a dataset (Ribeiro et al., 2020, Nat Commun; Strodthoff et al., 2021, IEEE JBHI on PTB-XL), and the PhysioNet/CinC Challenges 2020-2021 (Perez Alday et al., 2020, Physiol Meas; Reyna et al., 2021, CinC) explicitly pooled recordings from seven institutions in four countries to force generalisation. Yet reported drops when a model trained on one source is evaluated on another vary from negligible (atrial fibrillation) to large (conduction and repolarisation abnormalities), and nobody has systematically explained *why*: the sources differ simultaneously in (i) the label vocabulary and annotation practice (SCP-ECG statements in PTB-XL vs SNOMED-CT in Chapman/Ningbo/Georgia vs six free-text-derived classes in CODE-15% vs machine reports in MIMIC-IV-ECG), (ii) acquisition (400 vs 500 Hz, different filters and vendors), and (iii) the population (age structure, sex ratio, disease prevalence, ancestry). ECG foundation models (ECG-FM, McKeen et al., 2024/JAMIA Open 2025; HuBERT-ECG, Coppola et al., 2024, medRxiv; ECGFounder, Li et al., 2024, arXiv:2410.04133) promise robustness but have been benchmarked mainly on pooled random splits.

## The research gap

**What has been done (2020-2026):**

- PTB-XL benchmark and its cross-validation protocol (Wagner et al., 2020, Sci Data; Strodthoff et al., 2021, IEEE JBHI).
- CinC 2020/2021 Challenge: pooled multi-source training, hidden test sets from other sources; the Challenge reports overall generalisation but not a decomposition of the drop (Perez Alday et al., 2020; Reyna et al., 2021).
- CODE-15% and the CODE-II release (Ribeiro et al., 2021, Zenodo; Nat Digit Med 2026 CODE-II paper) show that Brazilian pre-training transfers to PTB-XL and CPSC-2018 after fine-tuning; no zero-shot population-shift analysis.
- OpenECG (arXiv:2503.00711, 2025) benchmarks self-supervised ECG models on 1.2M public records with leave-one-dataset-out experiments, but reports aggregate scores only, without label-mapping sensitivity, subgroup parity or clinical-outcome validation.
- "Benchmarking ECG FMs: a reality check across clinical tasks" (arXiv:2509.25095; ICLR 2026) evaluates eight foundation models on 26 tasks/12 datasets and finds gaps on outcome prediction, again on in-distribution splits.
- "Looking beyond accuracy: a holistic benchmark of ECG foundation models" (Filice et al., 2026, arXiv:2601.21830) studies embeddings across cross-continental datasets with SHAP/UMAP, but does not quantify subgroup parity or the contribution of label harmonisation.
- A 2026 Frontiers in Cardiovascular Medicine study (doi:10.3389/fcvm.2026.1928354) harmonised PTB-XL, Chapman-Ningbo and MIT-BIH into 13 SNOMED super-classes and showed feasible cross-population transfer with a compact feature model, while noting lower discrimination in patients >= 65 y and sex-dependent calibration; it used three sources, one mapping and no foundation models.
- Label ambiguity in PTB-XL is being addressed with partial-label learning (arXiv:2512.11095, 2025), and demographic-aware models report equalised-odds gaps across sex/age on Chapman (Sci Rep 2026, s41598-026-54206-8).
- Mortality prediction from ECG has been benchmarked on CODE-15 and MIMIC-IV-ECG with concordance drops of 0.03-0.24 on external validation (Lukyanenko et al., 2025), and MIMIC-IV-ECG-Ext-ICD (Strodthoff et al., 2024, Eur Heart J Digit Health) provides ICD labels aligned to ECGs.

**What is missing (the gap this project fills):**

1. No study *decomposes* the cross-source drop into label-mapping, acquisition and population components using controlled counterfactuals (same model, different mapping policy; same model, resampled/re-filtered signals; same model, age/sex-reweighted target).
2. No public leaderboard reports ECG transfer with **label-noise-aware** metrics (agreement-restricted evaluation, noise-rate sensitivity bands) although every source has a different annotation process.
3. Foundation models have not been compared *head-to-head with a simple supervised baseline on the same population-shift grid* with subgroup (age/sex) parity, i.e. whether pre-training buys robustness or merely average accuracy.
4. Diagnostic-label transfer has not been closed with a **clinical-outcome validation**: does the model that transfers best across countries also predict 1-year mortality and troponin elevation in MIMIC-IV-ECG, where linked EHR outcomes exist?

## Research questions / hypotheses

1. **RQ1 (decomposition).** For a fixed 1D-ResNet trained on source S and tested on target T, how much of the AUROC drop is removed by (a) using a lenient vs strict SNOMED mapping, (b) harmonising sampling rate/bandpass, (c) importance-reweighting T to S's age/sex distribution? *H1:* label mapping explains the largest share for conduction-block and repolarisation classes; population explains the largest share for AF and hypertrophy-related classes.
2. **RQ2 (population vs device).** Is the drop between two Chinese sources acquired with the same device family (Chapman vs Ningbo) smaller than between sources from different countries acquired with similar devices (PTB-XL vs Georgia)? *H2:* population drop > device drop after label harmonisation.
3. **RQ3 (foundation models).** Do frozen and fine-tuned ECG-FM / HuBERT-ECG / ECGFounder embeddings reduce the population-shift drop relative to a supervised 1D-ResNet at matched in-distribution AUROC? *H3:* fine-tuned foundation models reduce the average drop by >= 30 % but do not reduce subgroup parity gaps.
4. **RQ4 (label noise).** Do rankings of models change when evaluation is restricted to records whose labels agree across two independent mapping policies, or when label-flip rates are simulated at plausible levels (5-15 %)? *H4:* rankings among top models are unstable under noise for classes with kappa < 0.6 across mappings.
5. **RQ5 (parity).** Are age (>= 65 y) and sex AUROC gaps larger on out-of-source targets than in-source? *H5:* gaps widen out-of-source, and the widening is larger for the Brazilian (CODE-15%) target, which has the youngest population.
6. **RQ6 (clinical validation).** Among models trained on the harmonised diagnostic labels, does cross-country transfer performance correlate with prognostic performance for 1-year all-cause mortality and troponin-T elevation in MIMIC-IV-ECG? *H6:* Spearman rho > 0.5 between transfer AUROC (averaged over targets) and mortality C-index.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| PTB-XL v1.0.3 (Germany) | Source/target; SCP-ECG labels, age, sex | 21,799 ECGs, 18,869 patients, 500 Hz, 10 s | Open (PhysioNet, no credentialing) | https://physionet.org/content/ptb-xl/1.0.3/ |
| Chapman-Shaoxing + Ningbo (China) | Source/target; SNOMED labels, age, sex | 45,152 ECGs (10,646 Chapman + 34,905 Ningbo), 500 Hz, 10 s | Open (PhysioNet) | https://physionet.org/content/ecg-arrhythmia/1.0.0/ |
| Georgia 12-lead (USA) | Target; SNOMED labels | 10,344 ECGs, 500 Hz, 10 s | Open (PhysioNet Challenge 2021 training set) | https://physionet.org/content/challenge-2021/1.0.3/ |
| CPSC-2018 + CPSC-Extra (China) | Target; SNOMED labels | 6,877 + 3,453 ECGs, 500 Hz, 6-60 s | Open (PhysioNet Challenge 2021 training set) | https://physionet.org/content/challenge-2021/1.0.3/ |
| CODE-15% (Brazil) | Source/target; 6 classes, age, sex | 345,779 ECGs, 233,770 patients, 400 Hz, HDF5 | Open (Zenodo, CC-BY) | https://zenodo.org/records/4916206 |
| MIMIC-IV-ECG v1.0 (USA) | Outcome validation (mortality, troponin), machine-read reports | ~800,000 ECGs, ~160,000 patients, 500 Hz, 10 s | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| MIMIC-IV v3.1 hosp module | Death dates, labevents (troponin T itemid 51003), demographics | ~365k patients | Credentialed (PhysioNet) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV-ECG-Ext-ICD v1.0.1 | Optional ICD-10 labels aligned to ECGs | Labels for MIMIC-IV-ECG | Credentialed (PhysioNet) | https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/ |

Population contrast: PTB-XL median age ~61 y; Chapman-Ningbo ~51 y; CODE-15% younger primary-care telehealth population; MIMIC-IV-ECG is an in-hospital/ED population.

## Methods

1. **Harmonised loading** (`ecg_xgen.loaders`): every record is converted to a (12, 5000) float32 array at 500 Hz, standard lead order I, II, III, aVR, aVL, aVF, V1-V6, 10 s crop/zero-pad, 0.5-45 Hz Butterworth band-pass (configurable), per-lead z-scoring using *source* statistics only. CODE-15% (400 Hz, HDF5, 4096-sample zero-padded) is resampled with `scipy.signal.resample_poly(5, 4)`.
2. **Label harmonisation** (`ecg_xgen.labels`): a versioned mapping table from PTB-XL SCP codes, Challenge SNOMED codes, and CODE-15% columns to a target set of 12 harmonised classes (NORM, AF, AFL, 1dAVB, RBBB, LBBB, PVC, PAC, SB, ST, LQT, LVH) with three policies: `strict` (exact concept only), `lenient` (Challenge equivalence groups: CRBBB=RBBB, PAC=SVPB, PVC=VPB, includes IRBBB in RBBB), `superclass` (rhythm / conduction / morphology). Each policy is a deliberate experimental factor, not a nuisance.
3. **Baselines** (`ecg_xgen.models`): (a) handcrafted feature + logistic regression (RR statistics, QRS width proxy, spectral band powers) for a transparent lower bound; (b) 1D-ResNet-18 (PyTorch, optional), trained with BCE, AdamW, cosine schedule, 30 epochs, early stopping on source validation; (c) XResNet1d-101-style network from the PTB-XL benchmark as a stronger baseline (optional).
4. **Foundation models**: frozen-embedding linear probes and full fine-tuning for ECG-FM (open weights), HuBERT-ECG (open weights) and ECGFounder (open weights; verify licence). Identical harmonised inputs and splits.
5. **Transfer grid**: 5 sources x 5 targets (PTB-XL, Chapman, Ningbo, Georgia, CODE-15%; CPSC as an extra target) = 25 cells x 3 mapping policies x 3 model families. Also a leave-one-source-out pooled setting.
6. **Shift diagnostics** (`ecg_xgen.shift`): domain-classifier AUC (proxy A-distance) on handcrafted features and on penultimate embeddings; MMD with RBF kernel; per-lead amplitude/noise-floor and spectral fingerprints (to separate device from population); demographic-density-ratio weights (age, sex) fitted by logistic regression for reweighted evaluation.
7. **Counterfactual decomposition**: for each (S, T) cell compute AUROC under four conditions: raw; + harmonised mapping; + acquisition harmonisation (re-filter T with S's estimated filter, resample); + demographic reweighting. The sequential differences (Shapley-averaged over orderings) give the three components.
8. **Clinical-outcome validation**: fine-tune or linearly probe the harmonised-diagnosis models on MIMIC-IV-ECG for (i) 1-year all-cause mortality (Cox / discrete-time), (ii) troponin-T > 99th percentile URL within 24 h of the ECG, using the first ECG per hospitalisation; patient-level splits.

Tools: `wfdb`, `h5py`, `scipy`, `numpy`, `pandas`, `scikit-learn`, `torch` (optional), `lifelines` (optional, survival), `statsmodels` (optional).

## Evaluation & statistics

- Primary metric: macro AUROC over harmonised classes, plus per-class AUROC and AUPRC (prevalence differs across sources; AUPRC is reported alongside).
- Transfer drop: Delta = AUROC(in-source test) - AUROC(target). Decomposition via Shapley averaging over the 3! orderings of the three harmonisation operations.
- Label-noise-aware evaluation (`ecg_xgen.metrics`): (a) agreement-restricted AUROC on records whose harmonised label is identical under strict and lenient policies; (b) noise-sensitivity bands from simulated class-conditional flips at 5/10/15 %; (c) Cohen's kappa between mapping policies per class, reported with the leaderboard.
- Subgroup parity: AUROC and calibration (ECE, slope) by sex and age band (< 40, 40-64, >= 65); gap = max - min with stratified bootstrap CIs (2,000 resamples, patient-level). Equalised-odds gap at the source-selected operating point.
- Validation scheme: patient-level splits within each source (PTB-XL official folds 1-8/9/10; CODE-15% by patient_id; Chapman/Ningbo random patient split with seed); thresholds and normalisation statistics chosen on the source only. No target data touch training in the zero-shot setting.
- Leakage: patients appearing in both Chapman and Ningbo (same hospital system) are checked by hashing age/sex/date; CPSC and CPSC-Extra share provenance and are never split across train/test.
- Multiple comparisons: 25 cells x 12 classes; per-hypothesis Holm-Bonferroni within each RQ; report effect sizes with CIs rather than p-values as the main output.
- Nulls: (i) permutation of source labels within age/sex strata to obtain the null distribution of the "population" component; (ii) a label-shuffled model to establish chance-level transfer.
- Survival endpoints: Harrell's C and time-dependent AUROC at 1 year with IPCW; competing-risk-free because all-cause mortality.

## Publishable angle

- Headline: "Across four countries, label harmonisation explains X % of the apparent generalisation gap for conduction classes, population explains Y % for rhythm classes, and ECG foundation models close the average gap but not the age/sex parity gap; transfer quality predicts prognostic utility in MIMIC-IV-ECG."
- Deliverables: an open leaderboard (JSON + notebook), the versioned SNOMED mapping table with kappa statistics, and the decomposition code.
- Venues: *npj Digital Medicine*; *European Heart Journal - Digital Health*; *IEEE Journal of Biomedical and Health Informatics*; ML4H / CHIL (proceedings) for the benchmark paper.
- Follow-ups: single-lead / wearable transfer using the same harmonisation; adding CODE-II and MIMIC-IV-ECG as sources; test-time adaptation methods on the leaderboard; ancestry-aware analyses where self-reported race exists (MIMIC).

## Risks, confounds & mitigations

- **Label semantics differ beyond code mapping** (e.g. "sinus bradycardia" thresholds, machine vs cardiologist annotations in MIMIC). Mitigation: mapping policies as an explicit factor; agreement-restricted evaluation; report kappa.
- **Device and population are partially confounded** (each country uses its own vendors). Mitigation: Chapman vs Ningbo (same device family, same country) and PTB-XL vs Georgia (different countries, similar 500 Hz diagnostic carts) provide partial contrasts; spectral fingerprints quantify device effects.
- **Prevalence shift inflates/deflates AUPRC.** Report AUROC and AUPRC, and prevalence-matched subsampled AUPRC.
- **Foundation-model pre-training data overlap** with test sources (ECG-FM used some public data; HuBERT-ECG pre-trained on 9M ECGs including public sets). Mitigation: document overlap per model; exclude overlapping targets from zero-shot claims; treat as fine-tuned only.
- **CODE-15% has only six labels and zero-padded 7-10 s records.** Restrict CODE comparisons to the six shared classes; pad/crop consistently.
- **MIMIC-IV-ECG outcomes are confounded by indication** (ECGs ordered for sick patients). Use first ECG per admission, adjust for age/sex, report in ED-only subset.

## Milestones

- [ ] Download open sources (PTB-XL, Chapman/Ningbo, Challenge 2021 Georgia/CPSC, CODE-15%) with `scripts/download_data.py`; verify checksums.
- [ ] Freeze mapping table v1 (`ecg_xgen.labels`), compute per-class kappa between policies; write data card.
- [ ] Harmonised cache (.npy shards) per source; shift diagnostics report (domain-classifier AUC, spectral fingerprints).
- [ ] Handcrafted + logistic baseline on the 25-cell grid.
- [ ] 1D-ResNet on the grid, 3 seeds; decomposition experiment.
- [ ] Foundation-model probes and fine-tuning on the grid.
- [ ] Subgroup parity and label-noise-aware evaluation; bootstrap CIs.
- [ ] PhysioNet credentialing; MIMIC-IV-ECG + hosp outcome tables; mortality / troponin validation.
- [ ] Leaderboard release + manuscript.

## Ethics / data-use notes

- PTB-XL, Chapman/Ningbo, the Challenge 2021 training sets and CODE-15% are open (CC-BY / ODC-BY); cite the original papers.
- MIMIC-IV-ECG and MIMIC-IV are credentialed: complete CITI training and sign the PhysioNet DUA. Data must not be shared, must be stored encrypted at rest, and must **not** be sent to third-party LLM/API services except as permitted by PhysioNet's responsible-use policy. Never commit data, machine-read report text or derived per-record tables that could re-identify patients.
- Subgroup analyses are reported to expose, not to encode, disparities; do not use sex/age as model inputs in the main models (only for reweighting and evaluation).
- Foundation-model licences differ (research-only for some); check before redistribution of fine-tuned weights.
