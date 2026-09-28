# arterial-line-damping-detection

**Mine the fast-flush tests already recorded in open ICU and OR arterial waveforms to label under- and over-damping objectively at scale, train a flush-free damping detector, and quantify how damped lines distort hypotension flags, vasopressor titration and waveform-based machine learning.**

## Status / difficulty / timeline / compute

- Status: design + starter code (second-order catheter-transducer model with Gardner-style adequacy rules, a Windkessel waveform generator with a realistic upstroke, automatic detection of recorded flush tests and system identification of (fn, zeta) from the release, flush-free beat/spectral damping features with a grouped-CV classifier, and decision-impact tables: NIBP-ABP discrepancy, hypotension-threshold enrichment, event-rate ratios, cluster bootstrap). No data is shipped.
- Difficulty: MSc-level for the flush mining and detector; PhD-level with the decision-impact and ML-impact arms. 6-9 months.
- Compute: one workstation; MIMIC-III waveform records are processed one at a time with `wfdb`; the flush fit is a small grid search plus Nelder-Mead (about a second per event). No GPU required.

## Background

An arterial catheter, its tubing and transducer form a fluid-filled second-order system characterised by a natural frequency fn and a damping coefficient zeta (Gardner, Anesthesiology 1981). When zeta is too low for the available fn the system is under-damped ("resonant"): systolic pressure is overshot, dP/dt is exaggerated, and pulse-contour cardiac output is inflated. When zeta is too high, or fn collapses because of air bubbles, clots or kinks, the system is over-damped: the pulse is blunted, systolic pressure under-read and diastolic over-read. Mean pressure is nearly unaffected either way, which is why the harm is concentrated in decisions and algorithms that use systolic/diastolic values or pulse shape. The fast-flush (square-wave) test is the bedside way to check the system: flushing at 300 mmHg and releasing produces ringing whose frequency and decay give fn and zeta. Romagnoli et al. (Crit Care 2014) performed flush tests systematically in cardiovascular ICU patients and found abnormal damping in a substantial fraction of lines (roughly a third under-damped in that series), with clinically relevant systolic discrepancies; Saugel et al. (Crit Care 2020) made regular flush testing step one of their five-step approach to arterial pressure measurement.

Automated detection has been tried on small data: a J Clin Monit Comput 2021 study trained machine-learning models to recognise transducer-low, transducer-high and damped states from the waveform (AUCs 0.94-0.99 with patient-specific calibration); a recurrent CNN identified catheter damping during coronary angiography (JACC Cardiovasc Interv 2019); a 2024 paper moved "towards the automatic detection and correction of abnormal arterial pressure waveforms". Signal-quality indices for ABP in MIMIC (Sun, Reisner & Mark, Computers in Cardiology 2006; Li, Mark & Clifford, BioMed Eng OnLine 2009) treat flushes as artefacts to be excluded, and recent foundation-model work on signal quality in the critically ill (QualityFM, arXiv 2025) again learns generic artefact classes. The impact of damping on downstream measurements has been shown in small perioperative studies: under-damping and resonance filters change pulse-wave-analysis cardiac output (Br J Anaesth 2022) and inflate dP/dt max (Anesthesiology and Perioperative Science 2023).

## The research gap

What has been done (2014-2026):

- Prevalence of inadequate damping is known from single-centre series with study-team flush tests (Romagnoli et al., 2014) and from perioperative cohorts, never from open, multi-thousand-patient ICU waveform databases.
- ML detectors of damping states exist but were trained on labels the study team created (a few dozen patients), and none has been externally validated or released on open data.
- Every open ABP signal-quality pipeline discards flush artefacts; nobody has used the *recorded* flush tests as free, objective labels of the line's dynamic response.
- Downstream effects have been quantified for cardiac output and dP/dt in small OR cohorts; no study quantifies the effect of damping on the clinical labels and decisions taken from monitored pressure in the ICU (hypotension episodes, systolic-threshold triggers, vasopressor titrations), nor on the waveform-based models that the MIMIC waveform databases are used to train (cuffless BP, pulse-contour SV).

What is specifically missing (our angle):

1. **Flush mining as a labelling engine.** Detect the 300 mmHg square waves in MIMIC-III/MIMIC-IV/VitalDB arterial records and identify (fn, zeta) from each release by fitting the second-order model, giving thousands of objective damping labels per database without any new bedside test.
2. **Prevalence and dynamics at scale.** Fraction of line-hours under-/over-damped by database, unit, catheter site and dwell time; how damping drifts between flushes; whether flushes fix it.
3. **A flush-free detector** trained on flush-derived labels with record-grouped validation, transferred across databases and sampling rates (125 Hz MIMIC vs 500 Hz VitalDB), and released.
4. **Decision impact.** Enrichment of SBP<90 / MAP<65 minutes, of NIBP-ABP disagreement and of vasopressor rate changes in inadequately damped periods, with cluster-bootstrap CIs.
5. **ML impact.** Error of cuffless-BP (`cuffless-bp-mimic-waveform`) and pulse-contour SV (`pulse-contour-cardiac-output`) models stratified by damping class, i.e. how much of published waveform-model error is a line-quality problem.

## Research questions / hypotheses

1. **RQ1 (feasibility).** Are flush tests recoverable from the recordings? H1: >= 80% of MIMIC-III ABP records longer than 12 h contain at least one detectable flush with a fit r^2 >= 0.9; the median interval between flushes is < 8 h.
2. **RQ2 (prevalence).** H2: > 25% of labelled line-hours are inadequately damped (under-damping more common than over-damping in the ICU, the reverse during long OR cases as clots and air accumulate); over-damping increases with catheter dwell time.
3. **RQ3 (detector).** H3: window-level flush-free features distinguish adequate from inadequate damping with record-grouped AUROC >= 0.90 within database and >= 0.85 across databases without re-training; resampling VitalDB to 125 Hz costs < 0.03 AUROC.
4. **RQ4 (decision impact).** H4: over-damped minutes show an enrichment ratio > 1.3 for SBP<90 flags and under-damped minutes an enrichment > 1.3 for SBP>160 flags, with MAP<65 enrichment ~ 1.0; ABP-NIBP systolic bias exceeds +10 mmHg in under-damped and -10 mmHg in over-damped periods; vasopressor titration events are more frequent per hour in inadequately damped periods (rate ratio > 1.2).
5. **RQ5 (ML impact).** H5: cuffless-BP and pulse-contour errors are >= 50% larger in inadequately damped segments; excluding those segments improves reported model accuracy more than most architecture changes in the literature.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-III Waveform Database Matched Subset v1.0 | Primary ICU arm: ABP at 125 Hz with flushes; numerics (ABP Sys/Dias/Mean, NBP) | 22,317 waveform records, 10,282 patients | PhysioNet credentialed | https://physionet.org/content/mimic3wdb-matched/1.0/ |
| MIMIC-III v1.4 clinical | NIBP charted values, vasopressor infusions (`INPUTEVENTS_MV`), line insertion (`PROCEDUREEVENTS_MV`), ICU unit | ~46k ICU stays | PhysioNet credentialed | https://physionet.org/content/mimiciii/1.4/ |
| MIMIC-IV Waveform Database v0.1.0 | Pilot MIMIC-IV arm (ABP + ECG + PPG); larger releases expected | 200 records, 198 patients | PhysioNet credentialed | https://physionet.org/content/mimic4wdb/0.1.0/ |
| MIMIC-IV v3.1 `icu` | NIBP (220179/220180/220181), arterial BP charted (220050-220052), `inputevents` vasopressors, `procedureevents` arterial line | ~94k stays | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| VitalDB | OR arm: `SNUADC/ART` at 500 Hz with intra-operative flushes; NIBP; monitor-derived ART SBP/DBP/MBP | ~6,300 cases | Free registration (CC BY-NC-SA); REST API | https://vitaldb.net/dataset/ |
| Synthetic records (this repo, `transfer.py`) | Ground truth (fn, zeta) with embedded flushes for positive controls | any | Generated locally | - |

## Methods

Pipeline (each step maps to a module in `src/art_damping/`):

1. **Physics and controls** (`transfer.py`). Second-order catheter system (bilinear, pre-warped), `gardner_adequacy` (piecewise approximation of the adequacy chart, to be checked against the printed nomogram), a Windkessel generator with a fast skewed ejection and valve-closure backflow (normalised dP/dt ~ 20-25 /s, dicrotic notch), and `synthetic_flush` for square-wave tests through a given system.
2. **Flush mining** (`flush.py`). `detect_flush_events` finds plateaus > 250 mmHg lasting >= 0.2 s; `fit_flush_response` identifies (fn, zeta) and the fractional release time by fitting the second-order step-down response with a linear baseline (closed-form linear coefficients on a grid, Nelder-Mead refinement, r^2 reported); `flush_labels` and `propagate_labels` turn events into time-resolved adequacy labels with a maximum label age.
3. **Flush-free features** (`sqi.py`). Per beat: SBP, DBP, MAP, PP, HR, normalised dP/dt max, systolic width at 50% PP, systolic inflection count, dicrotic-notch prominence, systolic overshoot above the 5 Hz low-passed pulse, rise time, plausibility flag; per window: tilt-compensated spectral peakiness (resonant bump) and 8-25 Hz power ratio. `fit_damping_classifier` / `cv_auroc` train and evaluate with record-grouped folds.
4. **Decision impact** (`impact.py`). `pair_nibp_abp` and `discrepancy_by_class` (ABP-minus-NIBP bias/SD per class); `threshold_crossing_share` (share of SBP<90, MAP<65, SBP>160 minutes per class vs exposure share; enrichment ratio); `event_rate_ratio` for vasopressor rate changes; `cluster_bootstrap` by record.
5. **ML impact.** Apply the sibling projects' cuffless-BP and pulse-contour estimators to segments stratified by damping class; report error by class.
6. **Cohort covariates.** Catheter site and dwell time from `procedureevents`/`PROCEDUREEVENTS_MV`, unit type, vasopressor exposure.

### Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/download_data.py --simulate --out data/synthetic --n-records 20
```

```python
import numpy as np
from art_damping import transfer, flush, sqi
fs = 125.0
rec = np.load("data/synthetic/rec000.npz")
labels = flush.flush_labels(rec["abp"], fs)                     # fn, zeta, adequacy per recorded flush
beats = sqi.beat_damping_features(rec["abp"], fs)
windows = sqi.window_features(rec["abp"], fs, beats, window_s=60)
windows["damping_class"] = flush.propagate_labels(labels, windows["t_center"].to_numpy())
```

## Evaluation & statistics

- Ground truth: flush-derived (fn, zeta) with fit r^2 >= 0.9; positive control on synthetic records (tests require exact recovery); negative control: random 3.6-s segments without flushes must yield no events; sensitivity to the plateau threshold (200-280 mmHg) and label age (1-8 h).
- Detector: record-grouped 5-fold CV within MIMIC-III; external test on VitalDB (500 Hz and resampled to 125 Hz) and MIMIC-IV; AUROC/AUPRC with record-level bootstrap CIs; calibration of the predicted probability of inadequate damping.
- Decision impact: enrichment ratios and rate ratios with record-level cluster bootstrap (1000); NIBP-ABP Bland-Altman by class with repeated-measures variance; mixed-effects logistic models of threshold flags with random intercepts per record and covariates (vasopressor dose, unit, dwell time).
- Multiple comparisons: 3 thresholds x 3 classes x 3 databases; Benjamini-Hochberg; H1-H5 pre-registered.
- Leakage prevention: the classifier never sees the flush segment itself (windows containing a flush are excluded), and labels are propagated forward in time only.

## Publishable angle

Headline result: "Recorded fast-flush tests in 10,000 ICU arterial lines show that X% of monitored hours are inadequately damped; a flush-free detector recovers the state from routine beats; over-damped hours carry Y-fold more systolic hypotension flags and Z-fold more vasopressor titrations; and a large share of published waveform-model error occurs in those hours." The detector plus the labelled open cohort would be a standard preprocessing step for any ABP-based research.

Target venues: Critical Care Medicine or Intensive Care Medicine (prevalence and decision impact); Journal of Clinical Monitoring and Computing or British Journal of Anaesthesia (methods); IEEE Journal of Biomedical and Health Informatics (detector).

Follow-ups: correction (inverse filtering with the fitted system) and its validation against NIBP; damping-aware calibration of pulse-contour devices; a bedside alert prototype.

## Risks, confounds & mitigations

- Flush detection false positives (transducer zeroing, line disconnection spikes): require a plateau >= 0.2 s and a good fit (r^2 >= 0.9); inspect the distribution of plateau levels and durations.
- Steady-state assumption on the plateau: short flushes with very low fn/zeta are still ringing at release; the fit's r^2 and residual structure flag them; report sensitivity to minimum plateau length.
- The Gardner adequacy region is approximated piecewise; the breakpoints must be checked against the published chart and reported as a sensitivity analysis (two alternative boundaries).
- NIBP itself is imperfect and timing in chartevents is coarse: use +-90 s pairing, report both charted and monitor-numerics NIBP.
- Confounding of decisions by illness severity (sicker patients have more titrations and worse lines): adjust for vasopressor dose and unit, use within-record comparisons (before vs after a flush that restores adequacy).
- MIMIC-IV Waveform v0.1.0 is small: treat as pilot; re-run when larger releases appear.

## Milestones

- [ ] Synthetic-record pipeline end to end (`--simulate`, tests pass)
- [ ] VitalDB cases with ART + NIBP downloaded; flush detection tuned on OR data
- [ ] PhysioNet credentialing; MIMIC-III records with ABP selected; flush mining over the cohort; prevalence tables
- [ ] Flush-free detector with record-grouped CV; cross-database transfer
- [ ] Decision-impact analyses (thresholds, NIBP, vasopressor events) with cluster bootstrap
- [ ] ML-impact analyses with the sibling projects' models
- [ ] Pre-registration (OSF) before decision-impact analyses; manuscript and release of the detector

## Repository layout

```
README.md                         this document
requirements.txt                  numpy/scipy/pandas/scikit-learn, wfdb, vitaldb (optional), requests
data/README.md                    acquisition steps: synthetic records, VitalDB, MIMIC-III waveform + clinical, MIMIC-IV
scripts/download_data.py          --simulate (records with embedded flushes), VitalDB REST downloader, PhysioNet staging
src/art_damping/transfer.py       second-order catheter model, Gardner adequacy rules, Windkessel true ABP, synthetic flush
src/art_damping/flush.py          flush detection; (fn, zeta, release time) identification by model fitting; label propagation
src/art_damping/sqi.py            beat morphology + spectral damping features; window aggregation; grouped-CV classifier
src/art_damping/impact.py         NIBP-ABP discrepancy, threshold-crossing enrichment, event-rate ratios, cluster bootstrap
tests/test_art_damping.py         synthetic tests: SBP over/under-estimation, exact (fn, zeta) recovery, feature separation, impact tables
```

Analysis outputs (`outputs/`, git-ignored): per-record flush label tables, window feature tables with damping
class, prevalence tables, detector metrics, and decision-impact tables with bootstrap CIs.

## Ethics / data-use notes

- MIMIC-III, MIMIC-IV and their waveform databases are PhysioNet credentialed resources: CITI training, signed DUAs, approved encrypted storage; no redistribution; never send credentialed data to third-party LLM APIs or other external services.
- VitalDB is open under CC BY-NC-SA for research; cite Lee et al. (Sci Data 2022).
- Never commit data or derived record-level tables; `.gitignore` excludes `data/`, waveform files and `outputs/`. Publish aggregates only (counts >= 10).
- Credentials are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`; never hard-code them.
