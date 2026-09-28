# Echo-UQ: shift-aware conformal intervals for video-based ejection fraction across four echo datasets

**One-sentence pitch.** Test whether conformal prediction intervals for deep-learning left-ventricular ejection fraction (LVEF) stay valid when the model meets a new dataset (adult -> pediatric, apical 4- vs 2-chamber, vendor/site), whether the intervals flag physiologically hard studies (arrhythmia, poor image quality), and whether a physics-grounded Simpson's-biplane baseline computed from segmentation masks degrades more gracefully than end-to-end video regression.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data shipped; every echo dataset requires a registration or DUA.
- Difficulty: MSc to early-PhD (video deep learning + statistics of conformal inference + cardiac physiology).
- Timeline: 6-9 months (1 month data access + loaders, 2 months models + segmentation baseline, 2 months conformal experiments, 1-2 months hard-case analyses, remainder writing).
- Compute: one 24 GB GPU. Reproducing EchoNet-Dynamic's R(2+1)D on 10k videos takes ~1-2 GPU-days; DeepLabV3 segmentation ~1 GPU-day; the conformal and Simpson's analyses are CPU-only once predictions and masks are cached. Storage: EchoNet-Dynamic 7 GB, EchoNet-Pediatric ~12 GB, EchoNet-LVH ~9 GB, CAMUS 3 GB, TMED-2 ~2 GB.

## Background

EchoNet-Dynamic (Ouyang et al., 2020, Nature) showed that an R(2+1)D video network estimates LVEF from apical 4-chamber (A4C) clips with MAE ~4 % and released 10,030 videos with tracings. EchoNet-Pediatric (Reddy et al., 2023, JASE) trained a pediatric model on 4,467 studies (A4C and PSAX) and reported that adult-trained models are not applicable to children (and the reverse: R^2 0.33 on adults). CAMUS (Leclerc et al., 2019, IEEE TMI) provides 500 patients with A2C and A4C views, ED/ES masks, EF and *image-quality labels* (good / medium / poor). TMED-2 (Huang et al., 2022, DataPerf) provides 5,261 labelled 2D images with view labels (PLAX, PSAX, A2C, A4C, other) and aortic-stenosis severity, but no EF. EchoNet-LVH (Duffy et al., 2022, JAMA Cardiol) provides PLAX videos with wall-thickness measurements. Foundation models (EchoPrime, Vukadinovic et al., 2024; PanEcho, Holste et al., 2024/2025; EchoFM, 2024; EchoJEPA, 2026) report external LVEF MAE 4-5 % but give point estimates without calibrated uncertainty.

## The research gap

**What has been done (2020-2026):**

- Bayesian and ensemble uncertainty for EF: Kazemi Esfeh et al. (2020, MICCAI; "A deep Bayesian video analysis framework") and DEUE (2022, MICCAI) estimate epistemic uncertainty on EchoNet-Dynamic; EchoGNN (2022, MICCAI) gives explainable EF; uncertainty propagation through contour sampling (arXiv:2502.12713, 2025) for clinical metrics. None provides finite-sample coverage guarantees or tests validity under dataset shift.
- Cross-dataset segmentation shift: "Domain shift in echocardiography: interpretable quantification and prediction of cross-dataset LV segmentation" (arXiv:2607.19643, 2026) predicts Dice drop across six datasets from handcrafted / VAE descriptors; foundation-vs-domain-specific segmentation (npj Digit Med 2025, s41746-025-01730-y) reports EchoNet segmenters dropping 5-25 % on CAMUS/POCUS. These are about masks, not EF intervals.
- Diffusion regression for LVEF (MCSDR, arXiv:2602.08202, 2026) models a posterior over EF on EchoNet-Dynamic, EchoNet-Pediatric and CAMUS - a probabilistic model, but without conformal calibration or shift-validity tests.
- "The segmentation ceiling: why explicit LV masks do not improve learned EF regression" (arXiv:2609.19730, 2026) argues that masks add no accuracy in-distribution; whether mask-derived EF is more *robust* or better *calibrated* under shift is exactly the untested question.
- Weighted conformal prediction under covariate shift (Tibshirani et al., 2019, NeurIPS) and its extensions (Yang, Kuchibhotla & Tchetgen Tchetgen, 2024, JRSS-B; conformal predictive systems under covariate shift, arXiv:2404.15018) exist as statistical tools, but have not been applied to echo EF.

**What is missing (the gap this project fills):**

1. No study measures **empirical coverage of EF prediction intervals under dataset shift** (adult -> pediatric, A4C -> A2C, vendor/site), nor whether **weighted (shift-aware) conformal** restores it.
2. No study asks whether interval **width flags clinically hard cases** - atrial fibrillation / irregular RR (beat-to-beat EF variation), poor image quality (CAMUS labels), foreshortened views - which is what a clinician needs from "uncertainty".
3. No head-to-head comparison of **end-to-end regression vs segmentation-derived Simpson's biplane** on *interval validity and shift robustness* (the segmentation-ceiling paper compares accuracy only).
4. No use of TMED-2 / EchoNet-LVH as **out-of-view negatives** to test that uncertainty grows on wrong-view or off-distribution inputs (abstention behaviour), a prerequisite for safe deployment.

## Research questions / hypotheses

1. **RQ1 (validity under shift).** With split conformal calibrated on EchoNet-Dynamic (adult A4C), what is the empirical coverage of nominal 90 % intervals on (a) EchoNet-Pediatric A4C, (b) CAMUS A4C, (c) CAMUS A2C, (d) EchoNet-Pediatric PSAX? *H1:* coverage falls below 80 % for (a), (c) and (d); (b) stays within 85-92 %.
2. **RQ2 (shift-aware conformal).** Do likelihood-ratio weights estimated from a domain classifier on video-level features (Tibshirani et al., 2019) restore coverage to >= 88 % while keeping width < 1.5x? *H2:* yes for covariate-type shifts (vendor, quality), no for label-shift-dominated pediatric transfer unless a small pediatric calibration set (n >= 200) is used.
3. **RQ3 (hard cases).** Are locally adaptive interval widths larger for studies with irregular RR (proxy for AF from beat-to-beat EF variability and, where available, report text), poor CAMUS quality, and low segmentation confidence? *H3:* width AUROC for poor-vs-good quality >= 0.70; widths increase monotonically with beat-to-beat EF SD.
4. **RQ4 (physics-grounded baseline).** Does Simpson's-biplane EF from a DeepLabV3 segmenter transfer with a smaller MAE increase and better-preserved coverage than end-to-end R(2+1)D under the same shifts? *H4:* segmentation EF has larger in-distribution MAE (+1-2 %) but < half the MAE increase under adult -> pediatric shift and better interval calibration.
5. **RQ5 (abstention).** Do intervals on wrong-view inputs (TMED-2 PLAX/PSAX, EchoNet-LVH PLAX) exceed the 95th percentile of in-distribution widths in >= 90 % of cases? *H5:* yes for the segmentation route (mask area collapse), only partially for end-to-end regression.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| EchoNet-Dynamic | Training + calibration (adult A4C), EF, ESV/EDV, tracings | 10,030 videos, 112x112, Stanford | Free registration + research DUA (Stanford AIMI) | https://echonet.github.io/dynamic/ |
| EchoNet-Pediatric | Adult -> pediatric shift; A4C and PSAX; EF | 7,643 videos (A4C + PSAX) from 4,467 studies, ages 0-18 | Free registration + DUA (Stanford AIMI) | https://echonet.github.io/pediatric/ |
| EchoNet-LVH | Out-of-view (PLAX) inputs; no EF | ~12,000 PLAX videos with LV wall/cavity measurements | Free registration + DUA (Stanford AIMI) | https://echonet.github.io/lvh/ |
| CAMUS | View shift (A2C vs A4C), vendor/site shift, ED/ES masks, EF, image-quality labels | 500 patients (450 train + 50 test), GE Vivid E95, Univ. Hospital St Etienne | Open after registration (Creatis) | https://www.creatis.insa-lyon.fr/Challenge/camus/ |
| TMED-2 | View-labelled 2D images (PLAX/PSAX/A2C/A4C/other), AS severity; abstention tests | 599 studies, 5,261 labelled images, Tufts Medical Center | Free registration + DUA (Tufts) | https://tmed.cs.tufts.edu/tmed_v2.html |

EF labels exist only in EchoNet-Dynamic, EchoNet-Pediatric and CAMUS; TMED-2 and EchoNet-LVH are used as unlabeled / wrong-view distributions.

## Methods

1. **Video loading** (`echo_uq.video`): AVI -> uint8 array (T, H, W, 3) via OpenCV (fallback: imageio), grey-scale conversion, resize to 112x112, temporal subsampling (period 2), clip sampling (32 frames), EchoNet normalisation statistics; CAMUS NIfTI/MHD sequences via SimpleITK; TMED-2 PNG stacks.
2. **Segmentation + Simpson's EF** (`echo_uq.simpson`): frame-wise LV masks (DeepLabV3-ResNet50 trained on EchoNet tracings + CAMUS masks); long axis from the mask's principal axis, apex/base detection, 20 disks, single-plane (A4C or A2C) and biplane (CAMUS A2C+A4C) volumes; ED/ES selection from the mask-area curve; beat-to-beat EF series for RR-irregularity proxies. Pure numpy; tested against a synthetic ellipsoid with analytic volume.
3. **End-to-end regression**: R(2+1)D-18 (EchoNet-Dynamic recipe; `scripts/train_r2plus1d.py`), plus an MC-dropout / 5-seed ensemble head producing a heuristic scale sigma(x) for locally adaptive conformal scores.
4. **Conformal prediction** (`echo_uq.conformal`): split conformal with absolute-residual scores; locally adaptive scores |y - f(x)| / sigma(x); weighted conformal (Tibshirani et al., 2019) with likelihood ratios from a logistic domain classifier on video descriptors (intensity histograms, sector geometry, frame rate, mask-area statistics, embedding PCA); coverage, width, conditional coverage by group, and worst-slab coverage.
5. **Shift diagnostics** (`echo_uq.shift`): image-level descriptors, domain-classifier AUC / proxy A-distance, MMD on embeddings, effective sample size of the weights, and a per-dataset "shift card".
6. **Hard-case labels**: CAMUS quality labels; beat-to-beat EF SD and RR irregularity from the mask-area curve (EchoNet-Dynamic videos are multi-beat); optional keyword extraction from EchoNet metadata where available; view-mismatch from TMED-2 labels.

Tools: `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `opencv-python-headless`, `torch`/`torchvision` (optional), `SimpleITK` (CAMUS), `pandas`.

### Shift conditions (calibration always on EchoNet-Dynamic VAL unless stated)

| ID | Target | Shift type | EF label | n (approx.) | Primary question |
|---|---|---|---|---|---|
| S0 | EchoNet-Dynamic TEST | none (in-distribution) | yes | 1,277 | sanity: coverage = nominal |
| S1 | EchoNet-Pediatric A4C | population (age, size, HR) | yes | ~3,000 | RQ1, RQ2, RQ4 |
| S2 | EchoNet-Pediatric PSAX | population + view | yes | ~4,000 | RQ1 |
| S3 | CAMUS A4C | vendor/site, resolution, frame rate | yes | 500 | RQ1, RQ2, RQ3 (quality) |
| S4 | CAMUS A2C | view | yes | 500 | RQ1, RQ4 (biplane) |
| S5 | TMED-2 PLAX/PSAX/A2C/A4C images | wrong view / still images | no | 5,261 | RQ5 abstention |
| S6 | EchoNet-LVH PLAX | wrong view | no | ~12,000 | RQ5 abstention |
| S7 | EchoNet-Dynamic, RR-irregular subset | rhythm (proxy) | yes | derived | RQ3 |

### Interval methods compared

| Method | Score | Calibration data | Guarantee | Module function |
|---|---|---|---|---|
| Split conformal | \|y - f(x)\| | source VAL | marginal, exchangeable | `split_conformal_quantile` |
| Locally adaptive | \|y - f(x)\| / sigma(x), sigma from clip/ensemble SD | source VAL | marginal | same + `sigma` |
| Weighted conformal | as above, weights p_target/p_source from descriptors | source VAL + unlabeled target | marginal under covariate shift | `weighted_conformal_quantile`, `density_ratio_weights` |
| Target-calibrated | as above | 200 labelled target studies | marginal on target | `split_conformal_quantile` on target subset |
| Segmentation route | Simpson EF from masks, sigma from beat-to-beat EF SD | source VAL | as above | `simpson.ef_from_mask_stack`, `beat_to_beat_ef` |

## Evaluation & statistics

- Point accuracy: MAE, RMSE, R^2, Bland-Altman limits of agreement vs clinician EF; classification AUROC for EF < 40 % and < 55 % (pediatric threshold).
- Interval validity: empirical coverage at nominal 80/90/95 % with Clopper-Pearson CIs; mean and median width; conditional coverage by EF tertile, view, dataset, quality label, age band; worst-slab coverage over the descriptor space.
- Shift-aware arm: coverage and width with weighted conformal; effective sample size; sensitivity to weight clipping.
- Hard-case detection: AUROC of interval width (and of sigma(x)) for poor-vs-good quality, high-vs-low beat-to-beat EF SD, wrong-view vs correct-view.
- Validation scheme: EchoNet-Dynamic official train/val/test split; calibration set = official val (n = 1,288) or a patient-disjoint subset; test never used for calibration. CAMUS: official 450/50 split with 10-fold CV of the segmenter; EchoNet-Pediatric: patient-level splits by study. Videos from the same patient never straddle calibration and test.
- Leakage prevention: segmenter trained only on EchoNet-Dynamic train + CAMUS train; conformal calibration statistics stored per source and never pooled with targets in the zero-shot arm.
- Multiple comparisons: 5 target conditions x 3 nominal levels x 2 models; report coverage with CIs and pre-registered primary contrast (adult -> pediatric A4C at 90 %); Holm within families.
- Nulls: label-permuted calibration (coverage should equal nominal by construction; widths explode) as a sanity check; random weights vs learned weights for the shift-aware arm.

## Publishable angle

- Headline: "Conformal 90 % intervals calibrated on adult A4C echoes cover only X % of pediatric and Y % of A2C studies; likelihood-ratio reweighting restores coverage for vendor and quality shifts but not for pediatric transfer without local calibration; interval width flags poor-quality and arrhythmic studies with AUROC Z, and segmentation-derived Simpson's EF is less accurate in-distribution yet more robust under shift."
- Deliverables: cached predictions, masks and descriptors for four datasets; a conformal-EF evaluation harness; shift cards.
- Venues: *Medical Image Analysis*; *IEEE Transactions on Medical Imaging*; *JACC: Cardiovascular Imaging* (clinical framing); MICCAI / MIDL (methods).
- Follow-ups: conformal risk control for EF < 40 % decisions; test-time adaptation with weighted conformal as monitor; extension to right-ventricular function and strain; prospective silent evaluation.

## Risks, confounds & mitigations

- **Label-shift vs covariate-shift.** Pediatric EF distributions and normal ranges differ; weighted conformal assumes covariate shift. Mitigation: report both; add a small target calibration set arm; use label-conditional conformal as a further comparator.
- **Different EF ground-truth methods** (EchoNet: biplane Simpson's by sonographer; CAMUS: expert Simpson's from A2C/A4C; pediatric: 5/6 area-length for some). Mitigation: document per dataset; treat as part of the shift; sensitivity with volumes when available.
- **No explicit AF labels** in EchoNet-Dynamic. Mitigation: RR irregularity proxy from mask-area curves; validate the proxy on CAMUS (regular rhythm) and any report text; do not claim AF, claim "rhythm irregularity".
- **Segmentation failures on pediatric / A2C views** confound the physics baseline. Mitigation: report mask-quality gates; measure how often the pipeline abstains.
- **Small target sets (CAMUS test = 50)**: coverage CIs are wide. Mitigation: use CAMUS full 500 with cross-fitting; pre-register primary contrasts on the larger pediatric set.
- **Data agreements** forbid redistribution: share code, descriptors and aggregated results only.

## Milestones

- [ ] Obtain EchoNet-Dynamic, EchoNet-Pediatric, EchoNet-LVH (Stanford AIMI), CAMUS (Creatis), TMED-2 (Tufts); run `scripts/download_data.py --verify`.
- [ ] Run the synthetic end-to-end smoke test (`--sample`) and the unit tests.
- [ ] Reproduce EchoNet-Dynamic EF regression and segmentation on the official split.
- [ ] Simpson's biplane pipeline on EchoNet + CAMUS; agreement with reported EF.
- [ ] Split conformal on EchoNet val; coverage on all targets (RQ1).
- [ ] Domain classifier + weighted conformal; ESS diagnostics (RQ2).
- [ ] Hard-case analyses: quality, RR irregularity, wrong view (RQ3, RQ5).
- [ ] Segmentation vs regression robustness comparison (RQ4); manuscript.

## Ethics / data-use notes

- EchoNet-Dynamic / Pediatric / LVH: Stanford AIMI research use agreement (non-commercial, no redistribution, no re-identification attempts). CAMUS: cite Leclerc et al. 2019; TMED-2: Tufts data use agreement, non-commercial.
- Data are de-identified but are still patient videos: store on encrypted institutional storage, never commit videos, masks or per-patient tables; publish only aggregate metrics and code.
- Pediatric data warrant extra care in reporting (age bands rather than exact ages).
- No external LLM/API services receive any data; any report-text processing is local.
