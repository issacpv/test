# BioShift: a unified, WILDS-style distribution-shift benchmark across ICU, ECG, EEG and echocardiography

**One-sentence pitch.** Compose four sibling projects - ICU prediction across MIMIC-IV/eICU/HiRID/AmsterdamUMCdb, 12-lead ECG classification across five countries, scalp-EEG seizure detection across pediatric/adult/neonatal cohorts, and echocardiographic EF regression across adult/pediatric/vendor domains - into one benchmark with a shared task/split/metric interface and *typed shift axes* (site, population, device, label protocol, temporal), so that claims about distribution shift in clinical ML ("calibration fails first", "population shift hurts more than device shift", "diagnostics predict loss", "robustness methods do not transfer") can be tested across modalities instead of one dataset at a time.

## Status / difficulty / timeline / compute

- Status: design + working harness (this repo). The specification, adapters, metrics, diagnostics, gap decomposition and run loop are implemented and tested end-to-end on synthetic data for all four task types; `python scripts/download_data.py --dry-run` produces a leaderboard without any data. Real domains plug in through the cache contract in `data/README.md`.
- Difficulty: PhD-level as a whole (four data-access processes, four preprocessing pipelines) but modular: each modality is an MSc-sized sub-project already scaffolded in `../icu-model-transportability`, `../ecg-cross-dataset-generalization`, `../cross-dataset-seizure-generalization` and `../echonet-ef-uncertainty`. This project owns the interface, the cross-modality statistics and the leaderboard.
- Timeline: 9-15 months (3 months data access + cache exports, 3 months baselines on all cells, 3 months robustness-method sweep, 2-3 months cross-modality analysis + release, remainder writing).
- Compute: CPU for the harness, diagnostics and tabular/feature baselines; one 24 GB GPU for the ECG/EEG deep baselines and embeddings; one A100-class GPU for echo video models and foundation-model probes. Storage ~350 GB raw, caches < 5 GB.

## Background

WILDS (Koh et al., 2021, *ICML*; Sagawa et al., 2022, *ICLR*) showed that distribution-shift benchmarks with realistic, *typed* shifts change conclusions: methods that win on synthetic domain-generalisation suites (DomainBed; Gulrajani & Lopez-Paz, 2021, *ICLR*) rarely beat ERM in the wild. Its only clinical dataset is Camelyon17 (hospital shift in histopathology; Bandi et al., 2019, *IEEE TMI*). Clinical ML has documented the phenomenon repeatedly - chest X-ray models learning hospital signatures (Zech et al., 2018, *PLoS Med*), COVID models failing on external data (Roberts et al., 2021, *Nat Mach Intell*), dataset shift as a clinical safety issue (Finlayson et al., 2021, *NEJM*) - and has a clean vocabulary for shift types (Subbaswamy & Saria, 2020, *Biostatistics*; Lipton et al., 2018, *ICML* for label shift; Rabanser, Günnemann & Lipton, 2019, *NeurIPS* for detection). What it lacks is a benchmark in which the *same* evaluation is applied across signal modalities and the shift *type* is a variable.

## The research gap

**What has been done (2021-2026):**

- Clinical shift benchmarks are single-modality: BEDS-Bench (Avati et al., 2021, arXiv) for EHR OOD behaviour on MIMIC-III/eICU; EHRSHOT (Wornow et al., 2023, *NeurIPS D&B*) for few-shot EHR at one site; TableShift (Gardner, Popovic & Schmidt, 2023, *NeurIPS D&B*) for tabular shift including several clinical tasks; YAIB (van de Water et al., 2024, *ICLR*) for ICU tasks across MIMIC-IV/eICU/HiRID/AUMCdb; ManyDG (Yang, Westover & Sun, 2023, *ICLR*) treating each patient as a domain for EEG/EHR tasks; SubpopBench (Yang et al., 2023, *ICML*) for subpopulation shift including chest X-ray.
- Foundation-model benchmarks per modality report leave-one-dataset-out transfer: OpenECG (2025, arXiv:2503.00711), "Benchmarking ECG FMs: a reality check" (*ICLR 2026*), EEG-FM-Bench (*ICML 2026*, arXiv:2508.17742), OmniEEG-Bench (2026, arXiv:2606.00815), RobustSeiz (2026) for seizure detection; none types the shift or reports calibration/subgroup metrics uniformly.
- Shifts / Shifts 2.0 (Malinin et al., 2021, *NeurIPS D&B*; 2022) include medical segmentation (MS lesions) but no physiological time series or EHR.
- The sibling projects each decompose or diagnose shift *within* one modality (ICU: covariate/label/concept decomposition; ECG: label-mapping/device/population counterfactuals; EEG: montage vs age; echo: uncertainty under vendor/age shift).

**What is missing (this project):**

1. **A common interface** (`TaskSpec`, `DomainSpec`, `ShiftCell` with `axes`) and one harness so that a cell in ICU and a cell in EEG produce the *same* row schema: in-domain and transfer primary metric with group-bootstrap CIs, calibration intercept/slope/ECE, subgroup gaps, domain-classifier AUC, MMD, and the covariate/label/concept decomposition.
2. **Shift axes as a first-class variable** with contamination flags (foundation-model pre-training overlap), enabling cross-modality questions no single-modality benchmark can ask.
3. **Cross-modality tests** of four widely repeated but never jointly tested claims (H1-H4 below) with a mixed-effects model treating modality as a random effect.
4. **A method sweep under one protocol**: importance weighting, group DRO (Sagawa et al., 2020, *ICLR*), test-time normalisation adaptation, temperature/intercept recalibration with n = 100 target labels, evaluated on 70 cells across four modalities with the WILDS-style question "does anything beat ERM consistently?".

## Research questions / hypotheses

1. **H1 (calibration first).** In >= 75% of cells across all four modalities, the calibration-in-the-large shift (|intercept| on the target) exceeds its in-domain bootstrap CI while the primary discrimination metric's drop does not, i.e. calibration fails before discrimination regardless of modality.
2. **H2 (population > device).** Cells tagged `population` (age/case-mix: pediatric vs adult EEG and echo, CODE-15% vs European ECG, US vs European ICU) show a larger mean primary-metric drop than cells tagged `device` or `site` without `population`, within every modality; the axis effect is significant in a mixed model with modality random intercepts and slopes.
3. **H3 (diagnostics predict loss).** Domain-classifier AUC on model embeddings correlates with the primary-metric drop with Spearman rho >= 0.5 pooled over cells; the slope differs by modality (random-slope variance > 0), so a single diagnostic threshold cannot be transported.
4. **H4 (subgroup gaps widen).** Sex and age-band gaps in the primary metric are larger on targets than in-domain in >= 60% of cells and the widening persists after label-shift reweighting.
5. **H5 (no universal robustness method).** No method in the sweep improves the target primary metric beyond the in-domain bootstrap CI in more than two of four modalities; intercept recalibration with 100 target labels is the only intervention that improves calibration in all four.
6. **H6 (contamination).** For cells flagged `contaminated`, foundation-model transfer gaps are smaller than for clean cells by a margin exceeding the modality's in-domain CI, i.e. part of reported FM robustness is pre-training overlap.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 | ICU source/target (3 tasks) | ~94k stays | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| eICU-CRD v2.0 | ICU, multi-centre US | ~200k stays | Credentialed (PhysioNet) | https://physionet.org/content/eicu-crd/2.0/ |
| HiRID v1.1.1 | ICU, Switzerland | ~34k stays | Credentialed (PhysioNet) | https://physionet.org/content/hirid/1.1.1/ |
| AmsterdamUMCdb v1.0.2 | ICU, Netherlands | ~23k admissions | End-user licence (free) | https://amsterdammedicaldatascience.nl/amsterdamumcdb/ |
| PTB-XL v1.0.3 | ECG (Germany) | 21,799 ECGs | Open | https://physionet.org/content/ptb-xl/1.0.3/ |
| Chapman-Shaoxing + Ningbo | ECG (China) | 45,152 ECGs | Open | https://physionet.org/content/ecg-arrhythmia/1.0.0/ |
| Georgia + CPSC (Challenge 2021) | ECG (US, China) | 10,344 + 10,330 ECGs | Open | https://physionet.org/content/challenge-2021/1.0.3/ |
| CODE-15% | ECG (Brazil, 400 Hz) | 345,779 ECGs | Open (Zenodo CC-BY) | https://zenodo.org/records/4916206 |
| CHB-MIT | EEG source (pediatric) | ~980 h, 23 subjects | Open | https://physionet.org/content/chbmit/1.0.0/ |
| Siena Scalp EEG | EEG target (adult) | ~128 h, 14 subjects | Open | https://physionet.org/content/siena-scalp-eeg/1.0.0/ |
| TUSZ v2.0.3 | EEG target/source (mixed age) | ~1,600 h | Free registration (TUH DUA) | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ |
| Helsinki neonatal EEG | EEG target (neonatal) | 79 neonates | Open (Zenodo) | Stevenson et al., 2019, Sci Data |
| EchoNet-Dynamic | Echo source (adult, Stanford) | 10,030 A4C videos | Registration + research-use agreement | https://echonet.github.io/dynamic/ |
| EchoNet-Pediatric | Echo target (pediatric) | thousands of A4C/PSAX videos | Registration + agreement | https://echonet.github.io/pediatric/ |
| CAMUS | Echo target (France, GE Vivid E95, expert contours) | 500 patients | Free registration | https://www.creatis.insa-lyon.fr/Challenge/camus/ |

Cells: 36 ICU (3 tasks x 12 ordered pairs), 25 ECG (5 sources x 6 targets minus diagonal), 5 EEG, 3 echo = 69 cells in the v0 registry (`bioshift.spec.default_benchmark`).

## Methods

1. **Specification** (`bioshift.spec`): frozen dataclasses for tasks (type, labels, primary metric, group column, subgroup columns), domains (dataset, country, site, population, device, access) and shift cells (source, target, `axes`, `contaminated_models`); JSON round-trip; validation (unique ids, modality consistency, non-empty axes); `cell_table()` for the paper's cell inventory.
2. **Adapters** (`bioshift.adapters`): `load(task, domain, split) -> DomainData(X, y, groups, meta)`. `SyntheticAdapter` generates multi-domain data with per-domain covariate mean shift, label prior and concept rotation (deterministic per domain id) for every task type; `ManifestAdapter` reads the `.npz` caches and group-level split manifests exported by the sibling projects (contract in `data/README.md`).
3. **Metrics** (`bioshift.metrics`): AUROC/AUPRC/Brier/log-loss; calibration intercept (slope fixed at 1) and slope by weighted Newton logistic recalibration; quantile-bin ECE; multilabel macro versions; regression MAE/RMSE/R2/bias/slope with 50/90% coverage when a predictive SD is supplied; any-overlap event scoring with 30 s/60 s tolerances and FP/24 h for seizure windows (simplified SzCORE; the sibling's full scorer can be dropped in); subgroup max-min gaps; group bootstrap for the primary metric.
4. **Diagnostics** (`bioshift.diagnostics`): logistic domain-classifier AUC and proxy A-distance; RBF-MMD with permutation p; BBSE label-shift ratio; covariate/label/concept decomposition via importance re-weighting of the source evaluation, Shapley-averaged over the two orderings, with both orderings and the effective sample size reported.
5. **Harness** (`bioshift.harness`): `run_cell` fits the model on source train, evaluates on source test and every target, computes diagnostics and returns long-format rows; `run_benchmark`, `leaderboard`, `axis_summary`, `shift_loss_regression`, and `negative_control` (random group-disjoint halves of one domain; gap must be ~0).
6. **Models**: per-modality baselines from the siblings (LightGBM/LR for ICU; handcrafted+LR and 1D-ResNet for ECG; spectral+LR, Riemannian and compact CNN for EEG; the sibling's EF regressor for echo) wrapped in the `Model` protocol; foundation-model embeddings as `X` where available (ECG-FM, HuBERT-ECG; BIOT, LaBraM, EEGPT, CBraMod; EchoCLIP-style video encoders), with contamination flags.
7. **Robustness sweep**: ERM; importance-weighted ERM; group DRO over source subgroups; test-time batch-norm adaptation (deep models); CORAL on embeddings; intercept-only / temperature / Platt recalibration with n in {25, 100, 500} target labels (recalibration sample disjoint from evaluation).

Tools: `numpy`, `scipy`, `pandas`, `scikit-learn`; `statsmodels` (mixed models); modality-specific stacks live in the sibling projects.

## Evaluation & statistics

- Fixed group-level splits released as manifests (record ids only); in-domain metrics on the held-out source test split; targets evaluated in full; no target labels used for training, thresholds or normalisation except in the explicit recalibration arm.
- Uncertainty: group bootstrap (subject/patient/stay) with 200-2,000 resamples per cell; paired bootstrap for method-vs-ERM differences.
- Cross-modality claims (H1-H5): linear mixed models with cell-level outcomes (gap, |intercept|, subgroup-gap widening), fixed effects for axis indicators and method, random intercepts (and slopes for H3) by modality and by task; report estimates with 95% CIs; Benjamini-Hochberg across the 6 hypothesis families.
- Nulls and controls: `negative_control` per domain (gap ~0, domain AUC ~0.5); synthetic positive controls with known covariate/label/concept shift (`SyntheticAdapter` with `shift_scale`); label-permuted models for chance-level transfer; per-modality replication of each sibling's headline number as a sanity check that the interface did not change the result.
- Leakage: group-disjoint everything; Chapman/Ningbo provenance overlap and CPSC/CPSC-Extra handled as in the ECG sibling; FM contamination flags carried into every table.
- Multiple comparisons: primary hypotheses pre-registered (OSF); exploratory per-cell results shown as a leaderboard with CIs, no p-values.

## Publishable angle

- **Headline.** "Across 69 shift cells in four biomedical modalities, calibration drift precedes discrimination loss, population shift costs more than device or site shift, a single embedding-space diagnostic predicts loss with modality-specific slopes, and no robustness method beats ERM in more than two modalities; foundation-model gains shrink once pre-training contamination is excluded." Plus the benchmark itself: fixed manifests, cell registry, harness and leaderboard.
- **Venues.** *NeurIPS Datasets & Benchmarks* (benchmark paper); *Nature Machine Intelligence* or *npj Digital Medicine* (cross-modality findings); *ML4H* / *CHIL* (proceedings); *Scientific Data* for the manifest/cache release.
- **Follow-ups.** Temporal cells (MIMIC-IV anchor-year groups; TUSZ versions); imaging modalities (chest X-ray across MIMIC-CXR/CheXpert/NIH; MRI across OASIS/ADNI via the sibling projects); a "site-adaptation budget" estimator; a leaderboard service accepting only manifest-referenced submissions.

## Risks, confounds & mitigations

- **Axes are confounded within a cell** (a country change usually implies device and label-protocol changes). Mitigation: axis effects are estimated across many cells with mixed models; partial contrasts exist in each modality (Chapman vs Ningbo: same device/country; EchoNet-Dynamic vs Pediatric: same site/vendor; MIMIC vs eICU: same country); report axis effects as adjusted contrasts, not causal claims.
- **Feature/embedding choice changes diagnostics.** Mitigation: run diagnostics on both handcrafted features and model embeddings; report both; H3 pre-registered on embeddings.
- **Heterogeneous primary metrics** (AUROC, macro-AUROC, event F1, MAE). Mitigation: mixed models on *standardised* gaps (gap / in-domain bootstrap SD) and on ranks; sensitivity with AUROC-only where defined (EF < 40% binary for echo, window AUROC for EEG).
- **Data access asymmetry** (ICU credentialed vs open ECG/EEG). Mitigation: the open modalities form a fully reproducible public subset (`bioshift-open`); the ICU cells are reproducible by credentialed users from the same manifests.
- **The decomposition is not unique when shifts interact** (see the Brier example in the tests). Mitigation: report both orderings, Shapley average and ESS; treat components as descriptive.
- **Sibling pipelines evolve.** Mitigation: the cache contract is versioned; each cache file records the sibling commit hash in `meta`.

## Milestones

- [ ] Freeze registry v0 (`default_benchmark`) and JSON export; publish cell inventory.
- [ ] Cache exports from the ECG and EEG siblings (open data); manifests; `bioshift-open` leaderboard with feature baselines.
- [ ] Negative/positive controls on every open domain.
- [ ] ICU credentialing complete; ICU caches for 3 tasks x 4 sites; echo agreements complete; echo caches.
- [ ] Deep baselines and FM embeddings; contamination table.
- [ ] Robustness sweep on all cells; recalibration learning curves.
- [ ] Mixed-effects analysis of H1-H6; pre-registration filed before ICU/echo cells are evaluated.
- [ ] Release: manifests, harness v1.0, leaderboard JSON, paper.

## Ethics / data-use notes

- MIMIC-IV, eICU-CRD and HiRID are PhysioNet credentialed resources (CITI training + DUA); AmsterdamUMCdb has its own licence; TUSZ requires the TUH data-use form; EchoNet datasets require Stanford's research-use agreement; CAMUS requires registration. Data stay on approved storage; none may be redistributed.
- PhysioNet's responsible-use policy prohibits sending credentialed data (including derived per-patient tables or free text) to third-party LLM/API services; the harness is local-only and never uploads anything.
- Only manifests (public identifiers) and aggregate results (cell-level metrics, CIs) are released; no caches, embeddings or patient-level predictions.
- Subgroup analyses (sex, age band; race only where available and comparable) are descriptive audits of disparities, not model inputs.
- Credentials (`PHYSIONET_USER`, `PHYSIONET_PASS`, `TUH_USERNAME`, `TUH_PASSWORD`) are read from the environment only; `data/` and `outputs/` are git-ignored.
