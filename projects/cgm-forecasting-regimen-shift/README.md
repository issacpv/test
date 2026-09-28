# CGM-Regimen-Shift: does glucose forecasting break when the insulin-delivery regimen changes?

**One-sentence pitch.** Treat the *insulin-delivery regimen* (open-loop pump/MDI vs automated insulin delivery / DIY closed-loop) as an explicit distribution-shift axis for continuous-glucose forecasting, quantify how much accuracy and - critically - clinical hypoglycemia-warning performance degrade when a model trained under one regimen is deployed under another (across OhioT1DM, DiaTrend, AZT1D and the OpenAPS Data Commons), disentangle covariate shift from the controller-induced feedback that makes closed-loop glucose more predictable, and audit the DIY-loop data itself for the artefacts that inflate benchmark numbers.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No participant data are shipped.
- Difficulty: MSc-level (time-series ML + causal reasoning about feedback control + diabetes domain knowledge); PhD-level with the controller-feedback modelling.
- Timeline: 6-9 months (1-2 months harmonisation + regimen labeling + baselines, 2 months cross-regimen transfer grid + clinical metrics, 1-2 months feedback decomposition + data audit, remainder writing).
- Compute: CPU is enough for classical baselines and small LSTM/temporal-CNN models; one small GPU accelerates deep models. All datasets are small (MBs-GBs). Disk: <20 GB total.

## Background

Continuous glucose monitors (CGM) sample interstitial glucose every 5 minutes, and short-horizon forecasting (30-60 min ahead) underpins hypoglycemia alarms and decision support. The field has largely standardised on OhioT1DM (12 people with T1D, 8 weeks, open-loop pumps; Marling & Bunescu, 2020, CEUR / KHD workshop) as the shared benchmark, reporting RMSE at 30/60 min. But the therapeutic landscape has moved to **automated insulin delivery (AID)**: commercial hybrid closed-loop systems and the DIY OpenAPS/Loop community now dose insulin algorithmically in response to CGM. Under AID, the glucose trajectory is *the output of a control loop* - the very CGM signal being forecast is being actively regulated - which changes its statistics (less variance, controller-driven corrections) and, subtly, its predictability. A model trained on open-loop data (OhioT1DM) may transfer poorly, or *deceptively well*, to AID data, and vice versa.

Newer public datasets make the regimen axis analysable: DiaTrend (Prioleau et al., 2023, Sci Data, doi:10.1038/s41597-023-02469-5; 54 patients, mixed devices), AZT1D (2025, arXiv:2506.14789 / Mendeley; 25 patients all on Tandem Control-IQ AID with bolus detail and device mode), and the OpenAPS Data Commons (Lewis et al.; 100+ DIY-loop users, 46,000+ days). Recent benchmarks (MetaboNet-Bench, arXiv:2606.18640, 2026; Glucose-ML, arXiv:2507.14077, 2025) consolidate many datasets and CGM foundation models exist (GluFormer, Nature 2025, doi:10.1038/s41586-025-09925-9; CGM-LSM), but they optimise pooled or personalised accuracy and largely **do not treat regimen as a shift variable** with clinical-warning metrics.

## The research gap

**What has been done (2023-2026):**

- **Standard benchmark + models**: OhioT1DM RMSE leaderboards; personalized transformers (Gluformer, 2023); knowledge-distillation and glycemic-aware training (2025, arXiv:2502.14183).
- **Consolidated benchmarks**: MetaboNet/MetaboNet-Bench (2026) and Glucose-ML (2025) aggregate T1DEXI, DiaTrend, AZT1D etc.; CGM foundation models (GluFormer 2025; CGM-LSM with zero-shot OhioT1DM RMSE ~15.9 mg/dL for 1 h).
- **OpenAPS analyses**: large-scale glucose-variability and insulin-needs analyses on the OpenAPS Data Commons (e.g., temporal insulin-needs analysis, JMIR 2024, PMC11612593), and long-term forecasting for open-source AID (PMC10048652).
- **Safety critique**: "The Safety Challenges of Deep Learning in Real-World T1D Management" (arXiv:2310.14743, 2023) warns about distribution shift and clinical risk generally.

**What is specifically missing (the gap this project fills):**

1. No study frames **insulin regimen (open-loop vs AID/DIY-loop) as the explicit domain** and reports a cross-regimen transfer grid with both accuracy (RMSE/MAE) and **clinical hypoglycemia-warning** metrics (sensitivity/lead-time/false-alarm), which is where the harm lives.
2. No **decomposition** of the cross-regimen change into (a) covariate/marginal shift (AID glucose has lower variance/different autocorrelation) and (b) *controller-induced predictability* (the loop's corrective dosing is partly deterministic given CGM), separating "AID is easier to forecast" from "the model transfers."
3. No systematic **data-quality audit of the OpenAPS Data Commons** for artefacts that inflate benchmarks - CGM gaps/imputation, smoothing/interpolation, calibration jumps, duplicate/overlapping streams, algorithm-version heterogeneity - and their effect on reported RMSE.
4. No test of whether **regimen-aware features or normalization** (conditioning on device mode / active AID, variance-standardisation) restore cross-regimen warning performance, using AZT1D's device-mode labels.

This is complementary to the repo's clinical shift studies (`icu-model-transportability`, `ecg-cross-dataset-generalization`); here the shift is *therapy regimen* in T1D CGM.

## Research questions / hypotheses

1. **RQ1 (cross-regimen accuracy).** Training a 30/60-min forecaster on open-loop data (OhioT1DM) and testing on AID data (AZT1D, DiaTrend AID subset, OpenAPS), and the reverse, how large is the RMSE/MAE change vs within-regimen? *H1:* open-loop->AID RMSE *improves* on absolute scale (AID glucose is less variable) but the model is *miscalibrated* (systematic bias), while AID->open-loop degrades markedly.
2. **RQ2 (clinical warning).** How does hypoglycemia (<70 mg/dL) warning sensitivity, median lead time and false-alarm rate change across regimens at a fixed operating point? *H2:* cross-regimen transfer loses warning sensitivity and/or lead time even when RMSE looks fine - the clinically important failure.
3. **RQ3 (feedback decomposition).** How much of AID's apparent predictability is controller-induced (glucose partly determined by recent CGM through the loop)? *H3:* a model that also predicts the *controller's* next action explains a substantial share of AID glucose variance; conditioning forecasts on AID activity narrows the open-loop->AID calibration gap.
4. **RQ4 (data audit).** Do OpenAPS Data Commons artefacts (gaps, interpolation, calibration jumps, algorithm-version mix) materially change reported RMSE and the transfer conclusions? *H4:* naive inclusion of interpolated segments inflates apparent accuracy; artefact-filtered evaluation changes rankings.
5. **RQ5 (regimen-aware mitigation).** Do device-mode-conditioned features / variance-standardisation recover cross-regimen warning performance? *H5:* yes, partially - regimen-aware normalization closes much of the calibration gap but not the lead-time loss.

## Datasets

| Dataset | What is used | Size | Regimen | Access | URL |
|---|---|---|---|---|---|
| OhioT1DM (2018 + 2020) | CGM (5 min) + basal/bolus insulin + carbs + (some) HR/activity | 12 people, 8 wks each | Open-loop (pump/MDI) | DUA (request form) | http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html |
| DiaTrend | CGM + insulin-pump (basal/bolus) + carbs | 54 patients | Mixed (open-loop + some AID) | Free registration (Synapse) | https://doi.org/10.1038/s41597-023-02469-5 |
| AZT1D | CGM + pump + fine-grained bolus + **device mode** (regular/sleep/exercise) | 25 patients, 6-8 wks | AID (Tandem Control-IQ) | Open (Mendeley Data) | https://data.mendeley.com/datasets/gk9m674wcx/1 |
| OpenAPS Data Commons | CGM + insulin + algorithm decisions | 100+ users, 46,000+ days | DIY closed-loop (OpenAPS/AndroidAPS/Loop) | DUA (data request) | https://openaps.org/outcomes/ (data commons request) |
| T1DEXI (optional) | CGM + pump + wearable HR during exercise | 497 participants | Mixed | Free registration (Vivli/JAEB) | https://public.jaeb.org/ (T1DEXI) |

Regimen labeling: OhioT1DM = open-loop; AZT1D = AID (Control-IQ), with device-mode fields; DiaTrend/OpenAPS labeled per-stream from metadata (pump model / algorithm) into {open-loop, hybrid closed-loop, DIY-loop}.

## Methods

Pipeline (`src/cgm_shift/`):

1. **Harmonised loading & resampling** (`loaders.py`): parse each dataset's CGM + insulin + carb streams to a common 5-min grid; align event streams (bolus, basal-rate changes, carbs, device mode) as exogenous inputs; record data-quality flags (gap length, interpolation, sensor changes). Dataset-specific readers behind one interface; a synthetic generator ships for tests.
2. **Regimen labeling** (`regimen.py`): assign each stream/segment a regimen label and, for AID, an "AID-active" indicator from device mode / algorithm metadata.
3. **Forecasting models** (`models.py`): baselines - persistence (last value), linear AR / ARIMA-style, ridge on lagged features; deep - a small LSTM / temporal-CNN (PyTorch, optional). All produce 30- and 60-min-ahead predictions with optional exogenous inputs (insulin-on-board, carbs, device mode).
4. **Clinical metrics** (`clinical.py`): hypoglycemia (<70) and hyperglycemia (>180) event warning sensitivity, median detection lead time, false-alarm rate; Clarke/Parkes error-grid zone proportions; calibration bias by glucose range.
5. **Shift analysis** (`shift.py`): cross-regimen transfer grid; marginal-shift diagnostics (variance ratio, autocorrelation, MMD on windows); a controller-feedback decomposition that regresses AID glucose change on recent CGM+insulin to estimate the deterministic (loop-explained) fraction.
6. **Data audit** (`audit.py`): quantify gaps/interpolation/calibration jumps/algorithm-version mix per stream and re-evaluate under artefact-filtered vs naive inclusion.

Tools: `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels` (ARIMA, autocorrelation), optional `torch`.

## Evaluation & statistics

- **Accuracy metrics.** RMSE and MAE at 30 and 60 min; reported within-regimen and cross-regimen; time-lag-corrected RMSE to expose "predict-the-last-value" behavior.
- **Clinical metrics (primary for the harm claim).** Hypoglycemia warning sensitivity at a fixed false-alarm rate, median lead time (min), and Clarke error-grid Zone A+B proportion, all with bootstrap CIs (participant-level).
- **Validation scheme.** Participant-level splits; the official OhioT1DM train/test split preserved; leave-one-regimen-out for the transfer grid; personalization variants (fine-tune on the first days of a target participant) evaluated separately from zero-shot.
- **Leakage prevention.** No participant's data in both train and test; no future information in features (strict causal windows; insulin-on-board computed only from past doses); imputed/interpolated points flagged and excluded from the target of evaluation.
- **Feedback decomposition.** R^2 of AID glucose-change predicted from recent CGM+insulin (the loop-explained fraction); the "predictability" of AID vs open-loop compared at matched signal variance.
- **Multiple comparisons.** regimens x horizons x metrics x methods: Benjamini-Hochberg FDR within each RQ; effect sizes with CIs primary.
- **Nulls.** (i) Persistence and last-value baselines as the floor. (ii) Shuffled-participant transfer as chance. (iii) Variance-matched surrogate (rescale open-loop to AID variance) to test whether cross-regimen accuracy gains are just variance, not transfer.

## Publishable angle

- **Headline.** "Glucose forecasters trained on open-loop CGM look accurate on automated-insulin-delivery data by RMSE but lose hypoglycemia-warning sensitivity and lead time; much of AID's apparent predictability is controller-induced rather than learned transfer, and unaudited OpenAPS artefacts inflate benchmark accuracy - regimen-aware normalization recovers calibration but not lead time."
- **Deliverables.** A cross-regimen transfer benchmark with clinical-warning metrics; the controller-feedback decomposition; an OpenAPS data-quality audit and artefact-filtered evaluation protocol; regimen-labeling code and harmonised loaders.
- **Target venues.** *Journal of Diabetes Science and Technology*; *IEEE Journal of Biomedical and Health Informatics*; *npj Digital Medicine*; ML4H / CHIL; *Diabetes Technology & Therapeutics* for the clinical framing.
- **Follow-ups.** Add commercial hybrid-closed-loop datasets as they open; regimen-transition (peri-AID-initiation) forecasting; safety-aware training objectives that optimise warning lead time directly; foundation-model (GluFormer) zero-shot under the regimen grid.

## Risks, confounds & mitigations

- **Regimen is confounded with dataset/site/device** (each dataset is largely one regimen). Mitigation: use DiaTrend's within-dataset regimen mix and AZT1D's device-mode field for within-dataset contrasts; report device/site as nuisance; variance-matched surrogate null.
- **Small participant counts** (OhioT1DM n=12, AZT1D n=25). Mitigation: participant-level bootstrap; mixed-effects; report CIs and avoid over-claiming; pool where justified.
- **Data artefacts differ across datasets** and could drive "shift". Mitigation: the explicit audit (RQ4); artefact-filtered primary analysis; sensitivity to imputation policy.
- **"AID is easier" is not "model transfers".** Mitigation: the feedback decomposition and variance-matched surrogate separate predictability from transfer.
- **OpenAPS access latency / heterogeneity.** Mitigation: start with open AZT1D + OhioT1DM (DUA) + DiaTrend; add OpenAPS when the data request clears; algorithm-version recorded as a covariate.

## Milestones

- [ ] Request OhioT1DM (DUA) and OpenAPS Data Commons; download AZT1D (open) and DiaTrend (registration); harmonised loaders + tests.
- [ ] Regimen labeling + AID-active indicator; data-quality flags.
- [ ] Baselines (persistence, AR, ridge) + small deep model; within-regimen accuracy + clinical metrics.
- [ ] Cross-regimen transfer grid (accuracy + warning metrics) with participant bootstrap.
- [ ] Controller-feedback decomposition; variance-matched surrogate null.
- [ ] OpenAPS artefact audit; artefact-filtered re-evaluation.
- [ ] Regimen-aware normalization mitigation (RQ5); manuscript.

## Ethics / data-use notes

- OhioT1DM requires a signed DUA; OpenAPS Data Commons requires a data request and adherence to its use terms; DiaTrend and T1DEXI require registration; AZT1D is openly licensed on Mendeley Data - follow each licence and cite the dataset papers.
- These are de-identified free-living patient data; do not attempt re-identification. Store per-participant data locally; never commit data. Do not send participant data to third-party LLM/API services unless the dataset licence explicitly permits it.
- DIY-loop (OpenAPS) data come from a patient community using self-built systems; report findings respectfully and avoid implying the DIY systems are unsafe - the audit is about data artefacts for ML benchmarking, not about the therapy.
- Clinical-warning results are research analyses, not validated alarms; no clinical deployment is implied.
- Related projects in this repo: `icu-model-transportability`, `ecg-cross-dataset-generalization` - complementary transportability studies; this project is self-contained.
