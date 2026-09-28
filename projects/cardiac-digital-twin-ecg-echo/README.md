# Cardiac digital twins from unpaired ECG and echo: a physiology-anchored shared latent trained on PTB-XL + EchoNet, validated on paired MIMIC-IV-ECG/ECHO

One-sentence pitch: Learn a low-dimensional, interpretable cardiac state (EF, LV volumes, LV mass, conduction) that is jointly constrained by a large open ECG corpus (PTB-XL) and a large open echo corpus (EchoNet-Dynamic/LVH) even though the two never share a patient, then test on genuinely paired ECG-echo data (MIMIC-IV-ECG + MIMIC-IV-ECHO, EchoNext) whether the unpaired twin transports, is calibrated, and answers counterfactual questions that pure contrastive ECG-echo models cannot.

## Status / difficulty / timeline / compute

- Status: proposal + starter code (PTB-XL/EchoNet loaders, ECG feature extraction on a synthetic 12-lead generator, a forward "twin" model with unpaired quantile alignment and simulation-based regression, patient-level evaluation with cluster bootstrap).
- Difficulty: PhD-level (ML + cardiovascular physiology); 9-12 months to a first paper.
- Compute: one 24 GB GPU for ECG encoders and echo video encoders (EchoNet-Dynamic is ~7 GB of AVI); MIMIC-IV-ECHO DICOMs are several TB, so start with its derived measurements and a subset of clips. CPU-only is enough for the feature-level twin.
- Related projects (kept separate): `echonet-ef-uncertainty` (EF uncertainty from echo video), `ecg-cross-dataset-generalization` (ECG shift across PTB-XL/MIMIC-IV-ECG), `icu-model-transportability`.

## Background

ECG and echocardiography see the same heart through different physics: electrical activation and repolarisation versus mechanical geometry and motion. AI-ECG can screen for low ejection fraction (Attia et al., 2019, Nat Med) and structural heart disease (Poterucha et al., 2025, Nature; EchoNext), and echo video models estimate EF beat-to-beat (Ouyang et al., 2020, Nature; EchoNet-Dynamic). Cardiac "digital twins" (Corral-Acero et al., 2020, Eur Heart J; Niederer et al., 2021, Nat Rev Mater) promise a patient-specific state that explains all modalities at once and supports counterfactual questions, but mechanistic twins (e.g., ventricular activation twins fitted to 12-lead ECG and MRI: Gillette et al., 2021, Med Image Anal; Camps et al., 2024, Med Image Anal) need MRI and are built one patient at a time. The open data landscape is the opposite: huge unpaired ECG (PTB-XL, 21,799 records; Wagner et al., 2020, Sci Data; PTB-XL+ engineered features, Strodthoff et al., 2023, Sci Data) and echo corpora (EchoNet-Dynamic, 10,030 videos with EF/ESV/EDV; EchoNet-LVH, Duffy et al., 2022, JAMA Cardiol), plus a small number of credentialed paired sets (MIMIC-IV-ECG, Gow et al., 2023, PhysioNet; MIMIC-IV-ECHO, Gow et al., 2023, PhysioNet; EchoNext, 82,543 ECG-echo pairs from 36,286 patients, PhysioNet 2025).

## The research gap

What has been done (2024-2026):

- Paired cross-modal representation learning: EchoingECG (Gao, Kim & McIntosh, MICCAI 2025) learns probabilistic ECG embeddings aligned to echo with PCME++ and improves ECG-to-echo-label prediction; CardioState-JEPA (arXiv 2025/2026) learns a shared cardiac representation from MIMIC-IV-ECG with "delay-aware" handling of the ECG-echo acquisition gap; EchoJEPA (arXiv 2026) is an echo-only foundation model pre-trained on MIMIC-IV-ECHO; EchoBridge (arXiv 2026) aligns ECG, echo and text with a long-tail-aware objective; EchoCLIP (Christensen et al., 2024, Nat Med) aligns echo video with reports.
- ECG-only structural screening at scale (EchoNext, Nature 2025) with an open paired benchmark on PhysioNet.

What is specifically missing:

1. All joint ECG-echo models above require paired data from one institution and learn an opaque embedding. Nothing tests whether the two largest *open* corpora, which are unpaired and come from different countries and eras, can be fused into one state, and whether that state transports to paired data from a third institution (MIMIC/EchoNext). This is a transportability question the field has not asked.
2. No published ECG-echo model exposes an interpretable, physiologically constrained state on which counterfactuals can be run (e.g., "what would this ECG look like if EF were 35%?") and checked against known electro-mechanical relations (LV mass vs QRS voltage; dilation vs QRS duration; low EF vs T-wave/QRS patterns).
3. Calibration and discordance are unreported: when the ECG-derived state disagrees with the echo, is that flagged by the model's uncertainty, and is the disagreement informative (acquisition delay, pericardial effusion, conduction disease)?

Sharpened angle: an *unpaired* physiology-anchored twin (forward models + simulation-based inference + 1-D optimal-transport calibration of marginals) trained on PTB-XL and EchoNet, evaluated on paired MIMIC-IV-ECG/ECHO and EchoNext for (a) transport, (b) calibration, (c) counterfactual consistency, against paired contrastive baselines retrained on the same data budget.

## Research questions / hypotheses

1. H1 (unpaired feasibility): A twin trained without any paired sample predicts echo EF from ECG on paired MIMIC-IV test patients with MAE <= 9 EF points and AUROC >= 0.85 for EF < 40%, within 0.05 AUROC of a paired-trained baseline of the same architecture (non-inferiority margin pre-specified).
2. H2 (transportability): Performance drop from PTB-XL/EchoNet-domain validation to MIMIC-IV (ICU/ED population, different vendors) is smaller for the physiology-anchored twin than for an unconstrained contrastive embedding (difference in AUROC drop, paired bootstrap over patients).
3. H3 (counterfactual consistency): Intervening on the latent (EF -20 points; LV mass +50 g) moves generated ECG features in the physiologically expected direction in >= 80% of cases (QRS duration up, Sokolow-Lyon voltage up, respectively), a property paired embeddings do not have by construction.
4. H4 (discordance): The twin's predictive uncertainty (posterior width of EF given ECG) ranks ECG-echo discordant pairs (|EF_ecg - EF_echo| > 15) with AUROC >= 0.70, and discordance is enriched in pairs with > 30 days between acquisitions and in conduction disease (LBBB/paced), tested with logistic regression.
5. H5 (marginal-alignment validity): Quantile (1-D optimal-transport) alignment of the ECG-derived EF score to the EchoNet EF distribution is valid only if the two populations' EF distributions match; on MIMIC it must be re-estimated, and we quantify the bias when it is not (prior-shift sensitivity analysis).

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| PTB-XL 1.0.3 | Unpaired ECG corpus (train) | 21,799 12-lead ECGs, 18,869 patients, 10 s, 500/100 Hz | Open (CC BY 4.0) | https://physionet.org/content/ptb-xl/1.0.3/ |
| PTB-XL+ 1.0.1 | Engineered ECG features (QRS duration, axis, voltages) | features for all PTB-XL records | Open | https://physionet.org/content/ptb-xl-plus/ |
| EchoNet-Dynamic | Unpaired echo corpus (train): apical-4-chamber videos with EF, ESV, EDV | 10,030 videos | Free registration + Stanford AIMI DUA | https://echonet.github.io/dynamic/ |
| EchoNet-LVH | LV wall thickness / mass labels (PLAX) | 12,000 videos | Free registration + DUA | https://echonet.github.io/lvh/ |
| MIMIC-IV-ECG 1.0 | Paired evaluation (ECG side) | ~800k 12-lead ECGs, ~160k patients | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimic-iv-ecg/ |
| MIMIC-IV-ECHO 0.1 | Paired evaluation (echo side; DICOM clips, linkable by subject_id/hadm_id) | ~4.5k patients, >500k DICOM files | Credentialed | https://physionet.org/content/mimic-iv-echo/ |
| MIMIC-IV 3.x (hosp/icu) | Linking, timing, conduction-disease and pacing labels | full EHR | Credentialed | https://physionet.org/content/mimiciv/ |
| EchoNext | Large paired ECG-echo test set | 82,543 pairs, 36,286 patients | PhysioNet (verify current access tier on the page; expected credentialed) | https://physionet.org/content/echonext/ |

## Methods

1. ECG side (`cardiac_twin.ecg_features`): from raw 12-lead waveforms compute HR, QRS duration, QRS axis, Sokolow-Lyon voltage, plus a learned encoder (1-D ResNet as in PTB-XL benchmarks) producing a 64-d embedding; PTB-XL+ features are used as a check of the in-house extractor.
2. Echo side: EchoNet-Dynamic labels (EF, EDV, ESV) and, for videos, an EchoNet-style R2+1D encoder; EchoNet-LVH for wall thickness/mass.
3. Twin state z = (EF, EDV, LV mass, HR, QRS duration) with priors from population data (`cardiac_twin.twin_model.sample_prior`). Forward models: echo-side (identity + noise), ECG-side (linear-Gaussian in the starter code; a neural forward model later). Inference: simulation-based regression from ECG features to z, trained on samples from the prior + forward model, then marginal calibration to the unpaired echo cohort by 1-D optimal transport (`quantile_align`). A deep version replaces the linear forward with a conditional generative model trained with cycle/marginal-consistency losses; the starter code keeps the linear version so hypotheses H1/H3/H5 are testable early.
4. Paired baselines: (a) supervised ECG->EF regression trained on MIMIC pairs (upper bound), (b) contrastive ECG-echo (CLIP-style) retrained on the same MIMIC pairs, (c) zero-shot EchoingECG-style model if weights are public.
5. Pairing (`cardiac_twin.data.pair_ecg_echo`): nearest ECG within +/- 7 days of the echo (sensitivity: 1, 30 days); one pair per echo; patient-level splits.
6. Counterfactuals (`twin_model.counterfactual`): intervene on z and regenerate ECG features; score direction agreement (`evaluation.counterfactual_consistency`).
7. Tools: wfdb, neurokit2 (optional, cross-check of delineation), PyTorch, scikit-learn, pandas, pydicom for MIMIC-IV-ECHO.

## Evaluation & statistics

- Splits: PTB-XL official folds (1-8 train, 9 val, 10 test); EchoNet-Dynamic official split; MIMIC and EchoNext used only for testing (never for tuning the twin).
- Metrics: MAE / RMSE / Pearson r for EF; AUROC and AUPRC for EF < 40% and < 50%; calibration of predictive intervals (coverage at 50/90%, interval score); discordance AUROC (H4).
- Uncertainty: 95% CIs by patient-level cluster bootstrap (2,000 resamples); model comparison by paired bootstrap of the difference; non-inferiority tested against a pre-specified margin (H1).
- Multiple comparisons: five pre-registered hypotheses; secondary analyses Holm-corrected.
- Leakage prevention: patient-level splitting everywhere; MIMIC ECGs recorded after the echo of interest excluded from the "screening" analysis; acquisition delay reported.
- Nulls: label-permutation null for counterfactual consistency (expected 50%); prior-shift sensitivity by re-weighting EchoNet EF distribution to match MIMIC's.

## Publishable angle

Headline: "A physiology-anchored cardiac twin learned from unpaired open ECG and echo corpora transports to paired ICU data, supports counterfactual ECG generation, and flags ECG-echo discordance" - or the informative negative that unpaired fusion fails specifically on structural (LV mass) but not functional (EF) axes.

Target venues: Nature Machine Intelligence or npj Digital Medicine (main paper); MICCAI / MIDL (method); Circulation: Cardiovascular Imaging or JACC: Advances (clinical angle); IEEE TBME.

Follow-ups: add cardiac MRI (UK Biobank, application) as a third unpaired modality; replace the linear forward model with an electromechanical surrogate; longitudinal twins with serial ECGs in MIMIC-IV.

## Risks, confounds & mitigations

- Unpaired alignment is only identifiable up to distribution matching: use interpretable low-dimensional z, physiological forward constraints and report identifiability tests (H5). Report failure modes honestly.
- Population shift (PTB-XL is a German clinical cohort from 1989-1996; EchoNet is Stanford 2016-2018; MIMIC is Boston ED/ICU): treat as the object of study (H2), include age/sex adjustment.
- Acquisition delay between ECG and echo: model it (H4), restrict primary analysis to +/- 7 days.
- EchoNet EF labels are clinician-tracing based with ~5-7 point noise: propagate via the echo forward noise term.
- MIMIC-IV-ECHO measurements must be extracted from DICOM structured reports or derived by a video model: budget time; use `echonet-ef-uncertainty` tooling.
- Data-use: MIMIC/EchoNext data cannot be sent to third-party LLM/API services; all models local.

## Milestones

- [ ] Download PTB-XL (+ PTB-XL+) and EchoNet-Dynamic; validate ECG feature extractor against PTB-XL+ (QRS duration r > 0.9).
- [ ] Fit the linear twin; quantile-align to EchoNet EF; sanity-check counterfactuals (H3).
- [ ] Obtain PhysioNet credentials; build MIMIC-IV ECG-echo pairs (+/- 7 days); extract echo EF from DICOM SR / video model.
- [ ] Evaluate H1, H2, H4 on MIMIC; H5 sensitivity analysis.
- [ ] Deep twin (conditional generative forward model) and paired baselines; EchoNext evaluation.
- [ ] Manuscript, code and model release (weights trained on open data only).

## Ethics / data-use notes

- PTB-XL and PTB-XL+ are CC BY 4.0. EchoNet datasets require a Stanford AIMI research-use agreement; no redistribution.
- MIMIC-IV-ECG/ECHO, MIMIC-IV and EchoNext are credentialed PhysioNet resources (CITI training + DUA): keep on approved storage, do not upload to third-party LLM APIs except per PhysioNet's responsible-use policy, and never commit data or derived per-patient tables.
- Report demographic subgroup performance (sex, age bands, and race where recorded) for any screening claim.
