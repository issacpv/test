# Predicting Radiologist Disagreement in Lung Nodule Screening

**Inter-radiologist disagreement about lung nodule malignancy is predictable from the image. Treat it as a target rather than as noise: calibrate malignancy models on the readers' actual label distribution, predict where readers will disagree, and quantify what both do to Lung-RADS category assignment and screening false-positive rates.**

## Status / difficulty / timeline / compute

| | |
|---|---|
| Status | Design complete; starter code runs end-to-end on synthetic cohorts (70 tests) |
| Difficulty | MSc thesis, or a first PhD chapter. No exotic methods; the difficulty is in careful evaluation design |
| Timeline | 6–10 months. LIDC-only arms in 5; the Duke transfer arm adds 2–3 |
| Compute | A single workstation. Radiomics extraction over ~2600 LIDC nodules is a few CPU-hours; the models are logistic regression and gradient boosting, seconds to minutes. A GPU is needed only if the optional deep-learning comparison arm is included |
| Storage | ~200 GB (LIDC-IDRI ~125 GB, LUNA16 ~60 GB, Duke DLCS variable) |

## Background

Lung cancer screening with low-dose CT reduces lung-cancer mortality (Aberle et al., 2011, *NEJM*), and the screening enterprise now rests on a standardised reporting system: Lung-RADS assigns each screen a category from nodule type, size and growth, and each category maps to a follow-up interval (ACR, "ACR Lung-RADS v2022: Assessment Categories and Management Recommendations", *J Am Coll Radiol*, 2024). The size boundaries are sharp — 6 mm separates annual screening from a 6-month CT, 8 mm separates that from a 3-month CT or PET.

LIDC-IDRI (Armato et al., 2011, *Medical Physics*) is the dataset the whole nodule-analysis field was built on, and it is unusual in a way the field has mostly ignored: **four radiologists independently rated every nodule**, on a 1–5 malignancy scale, with no forced consensus. The resulting disagreement is enormous. Complete agreement on malignancy across all nodules in a case holds for only about 163 of roughly a thousand patients — disagreement on roughly 84%.

The standard response is to collapse it. LUNA16 (Setio et al., 2017, *Medical Image Analysis*) accepts nodules agreed on by at least three of four readers; malignancy classifiers typically train on the median rating binarised at 3. This is convenient and it is also a decision with consequences: it throws away a per-case measurement of how ambiguous the case is, and it trains models to be confident about cases where the evidence does not support confidence.

Meanwhile, the segmentation community took the opposite path. The Probabilistic U-Net (Kohl et al., 2018, *NeurIPS*) and PHiSeg (Baumgartner et al., 2019, *MICCAI*) model segmentation as a *distribution* over plausible annotations rather than a single mask, precisely because multiple expert delineations can all be clinically correct. Later work continued in this direction, including learning inter-rater variability with statistical distances (PULASki, arXiv:2312.15686, 2023) and rater-specific Bayesian networks. That machinery has stayed almost entirely in segmentation.

## The research gap

**What has been done.**

- **Label-noise handling on LIDC.** Several groups have recognised that aggregating reader disagreement hurts. "Re-thinking and Re-labeling LIDC-IDRI for Robust Pulmonary Cancer Prediction" (arXiv:2207.14238, 2022) proposes a divide-and-rule scheme to learn from ambiguous labels. Others have modelled the panel's ratings as a multi-label target, or predicted the panel's opinions directly. The framing throughout is **disagreement as a nuisance to be robust to**.
- **Distributional annotation models.** Probabilistic U-Net (Kohl et al., 2018, *NeurIPS*) and PHiSeg (Baumgartner et al., 2019, *MICCAI*) capture inter-rater variability in *segmentation*; a 2024 head-and-neck study found PHiSeg best for uncertainty estimation among the methods compared. These model mask distributions, not grading distributions, and are not evaluated against any management decision.
- **Calibration.** Modern classifiers are systematically overconfident (Guo et al., 2017, *ICML*), and post-hoc recalibration is routine. But calibration on LIDC is almost always measured against a *hard* aggregated label, which is the wrong reference when the label is genuinely probabilistic.
- **The Duke DLCS dataset** ("The Duke Lung Cancer Screening (DLCS) Dataset", *Radiology: Artificial Intelligence*, 2025, doi:10.1148/ryai.240248) supplies a contemporary, open screening cohort with 3D bounding-box annotations — a real external test set for screening-era CT, which LIDC (older scanners, non-screening indications) is not.

**What is specifically missing.** Four things, and the first is the crux:

1. **Disagreement has never been treated as a prediction target with an operational evaluation.** Everyone models malignancy and treats disagreement as noise. Nobody asks: *given the image, can we predict that four radiologists will split?* — and then evaluates that prediction the way it would actually be used, as a **second-read triage signal**. That reframing turns an R² into a recall-at-10%-review-budget, which is a quantity a screening programme can act on.
2. **No published comparison of soft-label versus majority-vote training that is scored against a common probabilistic reference.** The comparisons that exist score each arm against its own target, which is circular. Doing it properly requires a fixed reference soft label and identical folds — implemented here in `calibration.compare_label_schemes`.
3. **Nothing connects rating disagreement to Lung-RADS category disagreement.** Reader variance in arbitrary rating units is not interpretable to a clinical audience. "In N% of nodules, readers' own diameter estimates straddle a Lung-RADS size boundary, so the follow-up interval depends on which radiologist read the scan" is. The size boundaries are sharp and reader diameter estimates differ by more than a millimetre, so this number exists and has never been reported from LIDC's four-reader data.
4. **Uncertainty transportability is untested.** Does a disagreement model trained on LIDC still rank contemporary screening CT correctly? LIDC scanners and reconstruction kernels predate the screening era by a decade, so this is a real generalization question, and Duke DLCS makes it answerable for the first time.

**The angle in one sentence.** Multi-rater-aware calibration plus inter-rater-disagreement prediction, evaluated not by R² but by its effect on Lung-RADS category assignment, second-read triage yield, and false-positive rate at fixed sensitivity — extending the Probabilistic U-Net / PHiSeg idea from segmentation to malignancy grading and to screening operations.

## Research questions and hypotheses

1. **Is disagreement predictable at all?**
   *H1:* An image-feature model predicts per-nodule `binary_disagreement` with out-of-fold Spearman ρ > 0.4 and AUC > 0.70 for "readers crossed the benign/malignant boundary", under scan-level grouping.
2. **Does it beat the obvious baseline?**
   *H2 (the one that matters):* The model beats **threshold proximity** — `-min_t |diameter - t|` for Lung-RADS thresholds t ∈ {6, 8, 15} mm — by ΔAUC > 0.05. Nodules sitting on a size boundary are trivially disagreement-prone; a radiomics model that cannot beat "how close is this to 6 mm" has demonstrated nothing. *This is the baseline the label-noise literature does not report.*
3. **Triage yield.**
   *H3:* Reviewing the top 10% of nodules by predicted disagreement recalls ≥ 30% of all boundary-crossing disagreements, i.e. lift ≥ 3.0 over random review.
4. **Soft labels versus majority vote.**
   *H4:* Training on soft labels reduces expected calibration error by ≥ 30% relative to majority-vote training, scored against a common reference soft label on identical folds, **with no loss of AUC**. AUC parity matters: a calibration gain paid for with discrimination is not a win.
5. **Which soft-label convention?**
   *H5:* `split_3` (an indeterminate rating contributes 0.5) calibrates better than `exclude_3` (drop the 3s), because excluding indeterminate readers discards precisely the cases the model should be uncertain about.
6. **Lung-RADS consequences.**
   *H6:* In ≥ 10% of LIDC nodules with ≥ 3 readers, the readers' own diameter estimates produce different Lung-RADS categories, and in ≥ 5% they straddle the category-2/3 boundary — i.e. the follow-up interval depends on which radiologist read the scan.
7. **Operating points.**
   *H7:* At 95% sensitivity, deferring the highest-predicted-disagreement decile to a second read reduces the false-positive rate by ≥ 15% relative to no deferral.
8. **Transfer to Duke DLCS.**
   *H8:* A LIDC-trained disagreement model retains ≥ 60% of its LIDC Spearman ρ on DLCS when ranking a proxy uncertainty target. **Caveat that shapes the whole arm:** DLCS has no multi-reader ratings, so the target must be a proxy (e.g. annotation group, or malignancy-model entropy validated against diagnostic labels), and H8 is therefore weaker evidence than H1–H7. It is stated as exploratory.

## Datasets

| Name | What is used | Size | Access level | URL |
|---|---|---|---|---|
| LIDC-IDRI | 1018 thoracic CT cases; up to 4 readers per nodule with malignancy 1–5 plus 8 further semantic ratings and per-reader contours. The **only** source of the multi-reader malignancy target | ~125 GB DICOM; ~2600 nodules with ≥ 3 readers | **Open**, no registration (TCIA) | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| LUNA16 | Curated 888-scan LIDC subset (slice thickness ≤ 2.5 mm), official 10-fold split, detection candidates. Used for the false-positive-reduction arm and for comparability with published detection numbers | ~60 GB | **Open** (Zenodo) | https://zenodo.org/records/3723295 |
| Duke Lung Cancer Screening (DLCS) | 2061 screening LDCT scans, ~3187 semi-automatic nodule annotations (3D boxes); released subset 1613 volumes / 2487 nodules, with diagnostic labels. The **external screening-era transfer cohort** | tens of GB | **Open** (Zenodo) | https://doi.org/10.5281/zenodo.10782891 |
| NLST *(optional)* | Screening-scale cohort with outcomes, for realistic prevalence and operating-point context | large | **Application required** (NCI CDAS); weeks to approval; imaging is a separate request | https://cdas.cancer.gov/nlst/ |

**No credentialed (CITI-training) access is needed for the primary analysis.** LIDC-IDRI, LUNA16 and DLCS are all openly downloadable. NLST is optional and requires a data request.

## Methods

**Step 1 — annotation aggregation.** `annotations.load_pylidc_nodules` clusters each scan's annotations into nodules via `pylidc` and builds a `NoduleConsensus` per lesion, carrying every reader's malignancy rating, diameter and (optionally) mask. `build_nodule_table` emits one row per nodule with three soft labels, the majority vote, and eight disagreement metrics. Consensus masks come from `consensus_mask` at the 50% level, with the union/intersection spread retained via `segmentation_disagreement` as a segmentation-disagreement feature.

**Step 2 — inclusion and its cost.** Require ≥ 3 readers, following LUNA16. **Record how many nodules this excludes and report them separately**: a nodule that only one reader saw is the most uncertain kind there is, and dropping it silently biases the whole analysis toward agreement. This is a pre-registered reporting requirement, not an afterthought.

**Step 3 — features.** `features.extract_simple_features` gives 30 numpy/scipy features in four blocks: shape, attenuation, **boundary sharpness** (boundary gradient mean/sd/p90, inner-outer HU contrast, rim transition width) and GLCM texture, plus explicit **threshold-proximity** features. The boundary block is the substantive hypothesis: blurred, low-gradient margins are what readers disagree about. `extract_pyradiomics_features` is the reference backend (van Griethuysen et al., 2017, *Cancer Research*) with a fixed 25 HU **bin width** — a fixed bin *count* makes discretisation depend on each nodule's own intensity range and is a known reproducibility failure. Resample to 1 mm isotropic before texture (`resample_to_isotropic`); LIDC slice thickness spans 0.6–5 mm and the variation is correlated with scanner and era.

**Step 4 — malignancy model with soft labels.** `calibration.SoftLabelTrainer` with `route="sample_weighted"`: each nodule becomes a positive row of weight *p* and a negative row of weight *1−p*, optionally scaled by reader count. For log loss this is **exactly** soft-label cross-entropy, so it works with any scikit-learn estimator taking `sample_weight`. Baseline arm: `route="hard"` on the majority vote. All arms share folds and are scored against one common reference soft label (`compare_label_schemes`).

**Step 5 — disagreement model.** `disagreement.fit_disagreement_model` fits a grouped-CV gradient-boosting regressor on the same features against `binary_disagreement`. Evaluation is operational: `triage_curve` (recall, precision and lift at review budgets of 5–50%), `compare_to_size_baseline` (versus size and versus threshold proximity), and `selective_prediction_curve` (does deferring high-predicted-disagreement cases improve the malignancy model's accuracy?).

**Step 6 — Lung-RADS consequences.** `lungrads.assign_lung_rads` implements the v2022 baseline solid, part-solid and non-solid rules plus the new/growing rules, on **mean diameter** as the standard specifies. `category_disagreement` assigns a category from *each reader's own diameter* and reports whether they straddle a management boundary. `operating_point_analysis` reports sensitivity, specificity, FPR, PPV and a **prevalence-adjusted PPV** — essential, because PPV from a cohort where ~30% of nodules are malignant is meaningless for a programme where ~1% are.

**Step 7 — transfer.** Apply the frozen LIDC model to DLCS. Record the DLCS annotation group (1–4, by how much human confirmation each annotation received) as a covariate: it is a confounder for any uncertainty comparison, since groups 1–2 were annotated purely automatically.

**Baselines.** (a) Majority-vote hard-label classifier — the field standard. (b) Diameter alone for malignancy. (c) Threshold proximity alone for disagreement. (d) LIDC semantic ratings (subtlety, margin, spiculation) as features, to test whether radiomics adds anything over what a radiologist already writes down. Baseline (d) is the honest test of whether automated features are needed at all.

**Optional deep arm.** If time allows, compare a small 3D CNN trained with soft-label cross-entropy against the radiomics pipeline, and against a Probabilistic U-Net-style latent-variable head adapted to grading. This is the direct extension of Kohl et al. (2018) and Baumgartner et al. (2019) to malignancy, but it is a stretch goal — the radiomics version answers every hypothesis above.

## Evaluation and statistics

- **Primary outcomes:** (i) ΔECE between soft-label and majority-vote arms against a common reference; (ii) recall at a 10% review budget for the disagreement model, versus the threshold-proximity baseline.
- **Validation scheme:** 5-fold **GroupKFold on scan/patient ID**. This is non-negotiable and is the single most common error in the LIDC literature: nodules from one scan share the scanner, dose, reconstruction kernel *and* the same four readers. `tests/test_lidc_uq.py::test_nodule_level_split_leaks_relative_to_scan_level` demonstrates the inflation on synthetic data with planted scan-level structure.
- **Leakage prevention, specifically:** (i) group by scan, never by nodule; (ii) fit all feature scaling, selection and any calibration map inside the training fold; (iii) keep LUNA16's official folds when reporting detection numbers; (iv) DLCS is touched **once**, at the end, with frozen models — no iteration against it.
- **Calibration assessment:** ECE and MCE over equal-count bins, Brier against soft labels and against hard labels, and the Murphy three-way decomposition. The **uncertainty** term is reported prominently because it is the irreducible label-noise floor: a multi-rater dataset has a minimum achievable Brier score, and comparing raw Brier scores without it makes ambiguous data look like model failure.
- **Multiple comparisons:** the confirmatory family is H1–H7 (7 tests), Benjamini–Hochberg at q = 0.05. H8 and all per-subgroup breakdowns are exploratory and labelled as such.
- **Uncertainty quantification on the metrics themselves:** scan-level (cluster) bootstrap, 2000 replicates, for every reported metric and every paired difference. Resampling nodules rather than scans would understate every interval, since a scan contributes several correlated nodules.
- **Nulls:** (i) *label permutation* — shuffle disagreement targets within scan, confirming the model is not exploiting scan-level confounds; (ii) *feature permutation* — per-feature importance against a permuted-feature null; (iii) *reader permutation* — randomly reassign which reader gave which rating within a nodule, which leaves the soft label unchanged but destroys any reader-identity signal (LIDC readers are not consistently identified across cases, so a model must not depend on reader identity).
- **Sensitivity analyses:** minimum reader count ∈ {1, 2, 3, 4}; soft-label scheme ∈ {exclude_3, split_3, linear}; consensus level ∈ {union, 50%, intersection}; slice-thickness strata (≤ 1.5, 1.5–2.5, > 2.5 mm); with and without isotropic resampling; radiomics versus semantic-rating features.
- **Statistical tests:** paired scan-level bootstrap for differences in ECE, AUC and recall@k; DeLong or bootstrap for AUC differences; McNemar for category-assignment disagreement rates.

## Publishable angle

**Headline.** *"Radiologist disagreement about lung nodule malignancy is predictable from the image beyond what nodule size explains, and in N% of LIDC nodules the readers' own diameter estimates assign different Lung-RADS categories — so the follow-up interval depends on which radiologist read the scan. Training on the readers' label distribution rather than their majority vote reduces calibration error by X% at no cost in discrimination, and routing the top decile of predicted disagreement to a second read cuts the false-positive rate by Y% at fixed sensitivity."*

The Lung-RADS category-disagreement number is the figure that travels: it is a single interpretable statistic about a system in routine clinical use, derivable from open data, and as far as we can establish it has not been reported.

**Target venues.**
- *Radiology: Artificial Intelligence* — best fit; published the DLCS dataset, and the audience acts on Lung-RADS.
- *Medical Image Analysis* — if the methodological contribution (soft-label calibration, disagreement prediction) leads.
- *Medical Physics* — the journal of record for LIDC itself (Armato et al., 2011).
- *MICCAI* / *UNSURE* workshop — natural venue for the uncertainty-methods framing, and the right place for an early version.

**Follow-ups.** (1) Release per-nodule disagreement scores and soft labels for all of LIDC as a reusable derivative — a *Scientific Data* or dataset-descriptor paper on its own. (2) A prospective reader study: does showing a radiologist the predicted-disagreement flag change their read, and in which direction? (3) Extend to other multi-rater grading tasks (prostate PI-RADS, breast BI-RADS) to test whether the disagreement-prediction result is general. (4) Full distributional model: a latent-variable head that samples plausible rating *panels*, the direct grading analogue of the Probabilistic U-Net.

## Risks, confounds and mitigations

| Risk | Why it bites | Mitigation |
|---|---|---|
| **Disagreement may be mostly explained by size** | If threshold proximity beats the radiomics model, H2 fails and the novelty largely evaporates | This is the pre-registered primary comparison, not a robustness check. If size wins, that is itself the reportable finding: "reader disagreement in LIDC is a size-threshold artefact", which is a useful, publishable negative result and still supports the Lung-RADS analysis (H6) |
| **LIDC reader identities are not consistent across cases** | Any reader-level random effect or reader-bias model is unidentifiable | Never model reader identity across scans; the reader-permutation null makes dependence on it detectable. Treat `rater_id` as a within-scan index only, as documented in `RaterAnnotation` |
| **Nodule clustering is itself uncertain** | Readers sometimes disagree about whether there is one lesion or two; `n_raters < 4` conflates "a reader saw nothing" with "the clustering split the annotations" | Report the reader-count distribution; sensitivity analysis over the clustering tolerance; treat `n_raters` as a covariate and as an analysis stratum |
| **DLCS has no multi-reader ratings** | H8 cannot be tested directly | Stated as exploratory from the outset. Use a validated proxy target and be explicit that it is a proxy. Report DLCS annotation group as a confounder |
| **DLCS annotations are semi-automatic** | Groups 1–2 were produced by an algorithm, so "annotation uncertainty" there partly reflects the algorithm, not human readers | Stratify all DLCS results by annotation group; restrict the primary transfer analysis to the manually confirmed groups (3–4) |
| **Scanner and era confounding** | LIDC scanners predate screening-era CT; slice thickness correlates with scanner and with annotation practice | Isotropic resampling; slice-thickness strata; report scanner/manufacturer distributions; never let a model's apparent skill rest on acquisition signal |
| **Prevalence mismatch** | LIDC nodule malignancy prevalence is far above a screening population's | `operating_point_analysis(prevalence=...)` re-expresses PPV at screening prevalence; never report cohort PPV as though it were clinical PPV |
| **My Lung-RADS implementation is partial** | The S modifier, 4X upgrade and airway-nodule rules are not implemented | Documented explicitly in `lungrads.py`; unimplementable cases return category `"0"` rather than a guess. Have a radiologist review the implemented rules before publication, and report the fraction of nodules returning `"0"` |
| **Face-counting surface area in the numpy fallback** | Overestimates a sphere's area by ~50%, so fallback sphericity is not on the pyradiomics scale | Documented; use the pyradiomics backend for published numbers; never mix the two backends' features in one model |
| Small effective sample | ~2600 nodules across ~1000 patients, and scan-level grouping reduces the effective n further | Scan-level bootstrap everywhere; avoid deep models without strong regularization; do not chase subgroup effects |

## Milestones

- [ ] **M1 (month 1)** — LIDC-IDRI downloaded; `pylidc` configured; nodule table built; reader-count and disagreement distributions characterised; exclusion counts recorded
- [ ] **M2 (month 2)** — **H6 answered first**, because it needs only diameters: how often do readers' own measurements cross a Lung-RADS boundary? This is the cheapest high-value result and de-risks the project
- [ ] **M3 (month 3)** — Feature extraction over all nodules, both backends, with the isotropic-resampling comparison
- [ ] **M4 (month 4)** — Soft-label versus majority-vote comparison on common folds; H4 and H5 answered
- [ ] **M5 (month 5)** — Disagreement model; H1, H2, H3; triage and selective-prediction curves
- [ ] **M6 (month 6)** — Operating-point analysis and H7; all three nulls; full sensitivity grid; scan-level bootstrap CIs
- [ ] **M7 (month 8)** — Duke DLCS transfer arm (frozen models, single evaluation); H8 as exploratory
- [ ] **M8 (month 9–10)** — Manuscript; per-nodule soft labels and disagreement scores released as a derivative with code and exact software versions

## Ethics and data-use notes

- **LIDC-IDRI, LUNA16 and Duke DLCS are open, de-identified and require no application.** TCIA collections are released under a Creative Commons licence; cite the collection, the dataset paper (Armato et al., 2011, *Medical Physics*) and TCIA itself as the collection page specifies. Attribution is the data-use condition.
- **NLST requires an approved NCI CDAS data request.** Respect the terms of that agreement, which restrict redistribution and require acknowledgement of the NLST investigators. Do not include NLST-derived individual-level data in any released derivative.
- **Do not attempt re-identification.** These are de-identified CT volumes, but CT of the head or with skin surfaces can in principle support face reconstruction; LIDC is thoracic, so the risk is low, but the obligation stands. Never publish raw volumes.
- **Credentialed-data rule for the wider programme:** PhysioNet/MIMIC credentialed data may not be sent to third-party LLM APIs except as permitted by PhysioNet's responsible-use policy. **No dataset in this project is credentialed**, so that constraint does not bind here — but do not mix this repository's data with credentialed data, and do not paste clinical images into third-party services regardless of licence.
- **Never commit data.** `data/`, `outputs/` and `derivatives/` are gitignored, along with DICOM, NIfTI, MHD and archive extensions. Commit the download commands, the series manifest and the code. `~/.pylidcrc` is also gitignored: it contains a local absolute path, not a secret, but it is machine-specific and would break others' runs.
- **Clinical caution in framing.** A predicted-disagreement score is a research triage signal, not a diagnostic device. Any figure that flags nodules must be captioned as such, and the paper should state plainly that no prospective reader study has been done. Under-calling a malignant nodule is the asymmetric harm here, and every operating point reported should make the sensitivity explicit.
- **Radiologist involvement.** The Lung-RADS implementation and any clinical claim should be reviewed by a thoracic radiologist before submission. The rules encoded in `lungrads.py` are a partial implementation by a non-clinician and are documented as such.
