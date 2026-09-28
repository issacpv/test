# cuffless-bp-mimic-waveform

**A leakage-free audit of cuffless blood-pressure estimation from PPG/ECG: subject-independent, calibration-free evaluation across ICU (MIMIC) and operating-room (VitalDB) waveforms, with physiologically-informed PTT models vs deep learning judged on *BP-change tracking* around vasoactive drug boluses.**

## Status / difficulty / timeline / compute

- Status: design + starter code (streaming WFDB loader, signal-quality indices, beat/PTT feature extraction, subject-independent splitter with AAMI/ISO 81060-2 and IEEE 1708 evaluation, Moens-Korteweg baseline and a small 1D-CNN). No data shipped.
- Difficulty: MSc-level for the benchmark; PhD-level if the bolus-locked hemodynamic analysis is pushed to a mechanistic model.
- Timeline: 6-9 months.
- Compute: CPU for feature-based models; one mid-range GPU (8-16 GB) for the CNNs on ~1-5 M 10-s segments. Storage ~200-400 GB if full waveform records are cached locally (streaming avoids this).

## Background

Cuffless BP from photoplethysmography (PPG) alone or PPG+ECG (pulse arrival time, PAT; pulse transit time, PTT) is one of the most published topics in physiological signal processing. The physiology is well established: PTT relates to arterial stiffness through the Moens-Korteweg / Bramwell-Hill relations, and stiffness depends on transmural pressure (Mukkamala et al., 2015 IEEE TBME; Ding et al., 2016 IEEE JBHI). The ICU waveform databases (MIMIC-II/III, Moody & Mark) made large-scale training possible, and hundreds of deep-learning papers report mean absolute errors of 2-5 mmHg on MIMIC. Those numbers are mostly obtained with random segment-level splits in which the same patient (often the same minute) appears in training and test, i.e. the model learns the patient's BP level rather than BP physiology (Schrumpf et al., 2021 Sensors; Mukkamala et al., 2021 Hypertension on the evaluation problem; Elgendi et al., 2019 npj Digit Med).

PulseDB (Wang et al., 2023 Frontiers in Digital Health; 5.2 M 10-s ECG/PPG/ABP segments from 5,361 subjects drawn from MIMIC-III matched and VitalDB) was built to fix this: it provides subject-disjoint "calibration-free" and "AAMI" test sets and shows that calibration-free performance collapses (R^2 ~0.05 for SBP; MAE ~15 mmHg). A 2025 benchmarking study of deep PPG-BP models (arXiv 2502.19167, "Generalizable deep learning for photoplethysmography-based blood pressure estimation") confirms MAEs of ~12/8 mmHg (SBP/DBP) calibration-free vs ~9/6 calibrated, and a 2026 Scientific Reports paper on dataset integrity, calibration and signal quality reaches the same conclusion. Meanwhile IEEE 1708-2014 / 1708a-2019 and ISO 81060-3:2022 define how a cuffless device must be validated (subject counts, calibration intervals, induced BP changes) and are almost never used in ML papers.

## The research gap

Done by 2026:

- Leakage exposure: PulseDB (2023); Schrumpf et al. (2021) subject-independent PPG/rPPG; a 2023 study on data leakage in DL BP estimation; the 2025 benchmarking paper above; MIMIC-BP (Sci Data, 2024) curated subject-level splits.
- Physics-informed models: PITN (arXiv 2024, physics-informed temporal networks), BP-DeepONet (arXiv 2024), viscoelastic PTT extensions (arXiv 2026). These are evaluated on small wearable datasets or subject-dependent MIMIC splits.
- Change-focused evaluation: an August 2026 preprint ("Change point-aware evaluation and re-calibration of PPG-based BP estimation") evaluates tracking around statistically detected change points and shows models miss them. Change points are found in the ABP signal itself, so the analysis is not causally anchored and cannot separate "the model tracks the drug response" from "the model reacts to any waveform disturbance".
- Drug-challenge physiology: Finnegan et al. (2021 Sci Rep) and Payne et al. (2006 J Appl Physiol) showed in volunteers that PAT/PTT tracks BP changes for some vasoactive agents (angiotensin II, norepinephrine) but not others (GTN, salbutamol), i.e. the PTT-BP relation is drug-dependent because vasomotor tone changes the pre-ejection period and arterial compliance independently of pressure.

Missing (our angle):

1. A **cross-domain, subject-independent benchmark** trained on ICU waveforms (MIMIC-III matched subset, MIMIC-IV Waveform DB) and tested on the OR (VitalDB) and vice versa, with the *same* pipeline, SQI and split code, reporting AAMI/ISO 81060-2 and IEEE 1708 metrics and a "trivial predictor" floor (population mean; subject mean when calibration is allowed).
2. **Bolus-locked BP-change tracking**: instead of statistical change points, use *documented* vasoactive boluses — phenylephrine / ephedrine / norepinephrine pushes in MIMIC-IV `icu/inputevents` (ordercategorydescription "Drug Push"/"Bolus") linked to MIMIC-IV Waveform DB by `subject_id` and time, and infusion-pump tracks in VitalDB. The question is not "what is the SBP" but "did SBP go up by X mmHg in the 3 minutes after the bolus" — the quantity a closed-loop or alarm system needs. This is where physiology says PTT should work for alpha-agonists and DL might not.
3. **Hybrid physics vs pure DL**: a Moens-Korteweg PTT model (population-level and 1-point-calibrated), a feature ridge model, a small 1D-CNN and a hybrid (CNN residual on top of the PTT term), all under the same subject-independent evaluation, with per-drug-class stratification.
4. A **leakage taxonomy** with measured effect sizes: segment-level random split vs record-level vs subject-level vs subject+time-blocked, quantified on the same models.

## Research questions / hypotheses

1. **RQ1.** Under subject-independent, calibration-free evaluation, no model (PTT, ridge, CNN, hybrid) meets AAMI/ISO 81060-2 (ME <= 5, SD <= 8 mmHg, >= 85 subjects) for SBP on MIMIC or VitalDB; H1: SBP MAE >= 10 mmHg for all, within 3 mmHg of the population-mean predictor.
2. **RQ2.** Cross-domain transfer ICU -> OR degrades more than OR -> ICU (H2), because MIMIC's BP distribution is wider and its PPG SQI lower; the degradation is larger for the CNN than for the PTT-based models.
3. **RQ3 (change tracking).** In the 5 min after a phenylephrine/norepinephrine bolus, within-subject correlation between predicted and true delta-SBP is > 0.5 for PTT-based models and < 0.3 for the pure CNN (H3); after ephedrine (beta-agonist, raises HR/contractility) the PTT models' tracking drops (H3b, following Finnegan et al.).
4. **RQ4 (calibration interval).** With a single-point calibration per subject (IEEE 1708 "calibration" protocol), MAD drops below the IEEE grade B threshold (<= 6 mmHg) for <= 30 min and rises above grade D (> 7 mmHg) within 2 h in the ICU (H4), i.e. the calibration interval a wearable would need is shorter than what devices claim.
5. **RQ5 (leakage effect size).** Moving from segment-level to subject-level splits inflates SBP MAE by >= 6 mmHg for the CNN and by < 2 mmHg for the PTT model (H5), because the CNN memorises subject identity.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-III Waveform Database Matched Subset v1.0 | ICU training/test: ECG (II/V), PLETH, ABP at 125 Hz; linked to MIMIC-III clinical by subject_id | 22,317 waveform records, 10,282 patients (~2.4 TB full; stream only what has ECG+PLETH+ABP) | Open (PhysioNet, no credentialing for waveforms) | https://physionet.org/content/mimic3wdb-matched/1.0/ |
| MIMIC-IV Waveform Database v0.1.0 | ICU test set with **linkage to MIMIC-IV inputevents** (bolus timestamps) | 200 records / 198 patients (initial release) | Open on PhysioNet; linkage to MIMIC-IV v3.1 clinical tables requires credentialed access (CITI + DUA) | https://physionet.org/content/mimic4wdb/0.1.0/ |
| MIMIC-IV v3.1 `icu/inputevents`, `icu/icustays`, `hosp/patients` | Bolus events (phenylephrine 221749, norepinephrine 221906, epinephrine 221289, vasopressin 222315), demographics | ~9 M inputevents rows | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| VitalDB | OR test/training set: SNUADC/ECG_II, SNUADC/PLETH, SNUADC/ART at 500 Hz; drug tracks (Orchestra/* infusion pumps) and case metadata | 6,388 surgical cases | Open (CC BY 4.0) through the `vitaldb` Python package / REST API; dataset licence acceptance on vitaldb.net | https://vitaldb.net/dataset/ |
| PulseDB | Cross-check against published subject-disjoint splits ("CalibFree", "AAMI" test sets) | 5,245,454 10-s segments, 5,361 subjects (MIMIC-III + VitalDB) | Open (GitHub-hosted .mat v7.3 files) | https://github.com/pulselabteam/PulseDB |

## Methods

1. **Streaming extraction** (`wfdb_loader.py`): iterate PhysioNet records with `wfdb.rdrecord(pn_dir=...)` in chunks, keep records with an ECG lead + PLETH + ABP/ART, resample to 125 Hz, cut 10-s windows with 5-s hop, keep the record `base_datetime` and `subject_id` for clinical linkage. VitalDB through `vitaldb.load_case`.
2. **Signal quality** (`sqi.py`): PPG skewness / perfusion / template-correlation SQI (Elgendi, 2016 Bioengineering; Orphanidou et al., 2015 IEEE JBHI), ECG kurtosis / spectral SQI (Li, Clifford & Rajagopalan, 2008), ABP physiological-plausibility rules (Sun, Reisner & Mark, 2006 Computers in Cardiology "jSQI"-style checks). Windows are accepted only if all three pass; the acceptance rate per site is itself reported (it is a covariate-shift diagnostic).
3. **Beats and features** (`beats.py`): Pan-Tompkins-style R-peak detection; PPG foot / peak / max-slope fiducials; ABP beat SBP/DBP/MAP; PAT to foot/peak/max-slope, PPG-to-ABP PTT, HR, PPG amplitude, rise time, width; per-window medians and dispersion.
4. **Models** (`models.py`): (a) Moens-Korteweg PTT: SBP = a + b ln(PAT) (from E = E0 exp(alpha P)), population-fit and one-point-calibrated; (b) ridge on hand-crafted features; (c) 1D-CNN (3 residual conv blocks, global pooling, 2 outputs) on raw ECG+PPG; (d) hybrid = CNN residual added to the MK term with the MK parameters trained jointly.
5. **Splits** (`evaluation.py`): subject-level GroupKFold, plus time-blocked variants (train on the first 70% of each record, test on the last 30%) for the calibration-interval analysis, plus deliberately leaky segment-level splits for RQ5. Cross-domain: train on all MIMIC subjects, test on all VitalDB cases and vice versa.
6. **Bolus-locked analysis**: for each bolus, take the median SBP/DBP in [-3, -0.5] min and in [+1, +5] min; compute true and predicted delta; report within-subject correlation, sign agreement for |delta| >= 10 mmHg, and a mixed-effects slope (pred delta ~ true delta + (1|subject)). Boluses with < 60 s of accepted windows on either side are excluded.

### Window, feature and target definitions

| Item | Definition |
|---|---|
| Window | 10 s, 5 s hop, 125 Hz; must contain >= 6 matched beats and pass ECG, PPG and ABP SQI with PPG/ABP heart-rate agreement within 10 bpm |
| Reference SBP / DBP / MAP | Median over ABP beats in the window (peak, foot, beat mean) |
| PAT_foot / PAT_slope / PAT_peak | Median R-peak to PPG intersecting-tangent onset / max-slope point / systolic peak |
| PTT_abp_ppg | Median ABP onset to PPG onset (transit without pre-ejection period; only when the arterial line is proximal to the PPG site) |
| Morphology | HR, HR SD, PAT SD, PPG amplitude, rise time, width at 50% |
| Subject id | MIMIC: `subject_id` from the record name; VitalDB: `caseid`; PulseDB: `SubjectID` |
| Bolus event | MIMIC-IV `inputevents` rows with `ordercategorydescription` in ('Drug Push', 'Bolus') or duration <= 2 min for phenylephrine / norepinephrine / epinephrine / vasopressin; pre window [-3, -0.5] min, post window [+1, +5] min, >= 3 accepted windows each |

### Quickstart with the starter code

```python
import pandas as pd
from cuffless_bp import wfdb_loader, sqi, beats, evaluation, models

# stream windows from one open MIMIC-III matched record (no download)
wins = [w for w in wfdb_loader.iter_windows("p00/p000020/p000020-2183-04-28-17-47", "mimic3wdb-matched/1.0",
                                            max_windows=500)
        if sqi.window_quality(w.ecg, w.ppg, w.abp, w.fs)["accept"]]
df = beats.feature_table(wins)                       # one row per accepted window with PAT/PTT + SBP/DBP/MAP

# subject-independent evaluation of the physiological baseline
folds = evaluation.subject_independent_split(df["subject_id"], n_splits=5)
tr, te = folds[0]
evaluation.leakage_audit(tr, te, df["subject_id"])   # raises on any shared subject
mk = models.MoensKortewegPTT().fit(df["pat_foot"].iloc[tr], df["sbp"].iloc[tr], df["subject_id"].iloc[tr])
pred = mk.predict(df["pat_foot"].iloc[te])
print(evaluation.full_report(df["sbp"].iloc[te], pred, df["subject_id"].iloc[te]))   # AAMI/ISO, IEEE 1708, BHS, BA, tracking
floor = evaluation.trivial_baselines(df["sbp"].iloc[tr], df["subject_id"].iloc[tr], df["sbp"].iloc[te], df["subject_id"].iloc[te])
print(evaluation.aami_iso_evaluation(df["sbp"].iloc[te], floor["population_mean"], df["subject_id"].iloc[te]))

# bolus-locked change tracking (needs credentialed MIMIC-IV inputevents + MIMIC-IV WDB timestamps)
import duckdb
events = duckdb.connect().execute(wfdb_loader.mimiciv_bolus_sql("data/mimiciv")).df()
tbl = evaluation.bolus_response_table(df.assign(sbp_pred=pred_all), events)
print(evaluation.bolus_tracking_summary(tbl))
```

`tests/test_pipeline.py` builds synthetic ECG/PPG/ABP windows with a known PAT and checks that
the detector recovers HR within 2 bpm and PAT within 40 ms (currently ~11 ms), that the SQI
rejects flat, noisy and implausible windows, that leaky splits are caught, and that the
Moens-Korteweg fit recovers the pressure-stiffness slope from subject-centred data.

## Evaluation & statistics

- Primary: ISO 81060-2/AAMI (ME, SD of paired differences; both pooled and per-subject-averaged, >= 85 subjects with >= 3 measurements each), IEEE 1708 grade from MAD (A <= 5, B <= 6, C <= 7, D > 7 mmHg), BHS grade (cumulative % within 5/10/15 mmHg), Bland-Altman for repeated measures (Bland & Altman, 2007).
- Change tracking: within-subject Pearson r of deltas, concordance of direction, mixed-effects slope with 95% CI; per drug class.
- Floors and ceilings: population-mean predictor (calibration-free floor), subject-mean predictor (calibrated floor), ABP-derived "oracle" features (ceiling for the feature pipeline).
- Uncertainty: cluster bootstrap by subject (1000 resamples); paired comparisons between models on the same subjects.
- Nulls: subject-shuffled PPG (features from another subject) must give MAE at the floor; time-reversed PAT must destroy tracking.
- Multiple comparisons: Benjamini-Hochberg over models x metrics x domains; hypotheses H1-H5 pre-registered.
- Leakage prevention: `leakage_audit()` asserts subject disjointness and no temporal adjacency between train and test windows; all SQI thresholds and normalisation statistics fitted on training subjects only.

## Publishable angle

Headline: "Under subject-independent evaluation, cuffless BP models are within 2-3 mmHg of predicting the population mean; the only clinically useful signal is *change* tracking, and only the physiologically-constrained PTT models track alpha-agonist bolus responses (r > 0.5) while raw-waveform CNNs do not." Plus a cross-domain table (ICU <-> OR) and the leakage effect-size table.

Target venues: IEEE Transactions on Biomedical Engineering; Physiological Measurement; npj Digital Medicine; IEEE JBHI. Conference: IEEE EMBC / Computing in Cardiology for the benchmark release.

Follow-ups: extend to wearable-grade PPG (contactless/rPPG); model-based data assimilation (Windkessel + PTT) for continuous MAP; use the bolus-response data to fit subject-specific alpha (stiffness) parameters as a digital-twin calibration.

## Risks, confounds & mitigations

- Arterial-line damping/resonance corrupts the reference ABP: ABP SQI + exclusion of windows with abnormal dP/dt; sensitivity analysis on strict vs lenient SQI.
- PPG site differs (finger in both, but ear/forehead probes exist in the OR): use VitalDB track metadata; exclude non-finger probes.
- MIMIC-IV Waveform DB is small (198 patients); the bolus analysis will be limited to those with vasoactive pushes. Mitigation: MIMIC-III matched subset with MIMIC-III `inputevents_mv` gives more bolus events; VitalDB pump tracks give OR events.
- Time alignment between waveform records and `inputevents` timestamps is only as good as nurse charting (minute resolution, possible delays): use a [-3, +5] min window and a sensitivity analysis with +-2 min jitter; require an observable ABP change as a positive control for the alignment (but evaluate tracking on the model predictions only).
- Vasoactive boluses change vascular tone and PEP, so PAT-based models may track because of HR/PEP rather than stiffness: report PTT (PPG-ABP foot) alongside PAT to separate the PEP contribution.
- VitalDB and MIMIC differ in sampling rate and filters: resample to 125 Hz and apply identical band-pass filters; report a domain-classifier AUROC between the feature distributions.

## Milestones

- [ ] Streaming loader validated on 20 MIMIC-III matched records and 20 VitalDB cases; SQI acceptance rates reported
- [ ] Beat/PAT pipeline validated against ABP-derived oracle on synthetic and real windows
- [ ] Subject-independent benchmark table (4 models x 2 domains x AAMI/IEEE/BHS) with cluster-bootstrap CIs
- [ ] Leakage effect-size table (segment / record / subject / subject+time splits)
- [ ] Cross-domain transfer table (ICU -> OR, OR -> ICU)
- [ ] Bolus event table from MIMIC-IV inputevents linked to MIMIC-IV WDB; VitalDB pump events
- [ ] Bolus-locked change-tracking analysis per drug class
- [ ] Calibration-interval analysis (IEEE 1708) with time-blocked splits
- [ ] Pre-registration; manuscript + code/benchmark release

## Ethics / data-use notes

- MIMIC-III/MIMIC-IV waveform data are de-identified; the waveform subsets are open on PhysioNet, but linking to MIMIC-IV clinical tables (inputevents) requires credentialed access: CITI training, DUA, and compliance with PhysioNet's responsible-use policy, which prohibits sharing the data with third parties, including sending waveform or clinical records to third-party LLM APIs.
- VitalDB is CC BY 4.0; cite Lee et al. (2022 Sci Data) and respect the data-use terms on vitaldb.net.
- Scripts read `PHYSIONET_USER` / `PHYSIONET_PASS` from the environment for the credentialed parts; never commit credentials or data (`data/`, `*.dat`, `*.hea`, `*.mat` are git-ignored).
- Report results with the AAMI/ISO subject-count caveats; do not claim device-grade validation from retrospective data.
