# ventilator-asynchrony-detection

**From waveform-level asynchrony detection (a physics simulator that generates labelled asynchronies plus open waveform sets) to the charted footprint of asynchrony in HiRID, eICU-CRD and MIMIC-IV: a validated low-resolution asynchrony surrogate and its association with ventilation duration and mortality at scale.**

## Status / difficulty / timeline / compute

- Status: design + starter code (single-compartment lung and ventilator state machine that emits flow/pressure with ground-truth ineffective efforts, double, reverse and auto-triggering; breath segmentation and rule-based detectors; charted-resolution surrogate features including the monitor-minus-ventilator respiratory-rate discrepancy; surrogate calibration and evaluation; extraction templates and an exposure-outcome analysis with cluster-robust logistic regression). No data is shipped.
- Difficulty: MSc-level for the simulator/detector/surrogate chain; PhD-level for the three-database epidemiology with confounding control. 8-12 months.
- Compute: one workstation with >= 64 GB RAM for HiRID/eICU extraction (DuckDB over parquet/csv.gz); no GPU unless deep detectors are trained on the waveform sets.

## Background

Patient-ventilator asynchrony (PVA) is common and harmful. Thille et al. (Intensive Care Med 2006) found an asynchrony index (AI) above 10% in about a quarter of ventilated patients, associated with longer ventilation; Blanch et al. (Intensive Care Med 2015) showed with continuous automated monitoring that AI > 10% is associated with ICU and hospital mortality; Colombo et al. (Crit Care Med 2011) showed clinicians recognise asynchronies on the screen poorly. The event types have distinct waveform signatures: ineffective effort (an inspiratory effort that fails to trigger, seen as a notch in expiratory flow), double triggering (two breaths delivered on one effort, seen as a very short expiratory time between inspirations), reverse triggering (efforts entrained to mandatory breaths), auto-triggering (breaths without effort, e.g. from cardiac oscillations), and cycling asynchronies.

Automated detection began with rule-based systems on flow/pressure (Better Care; Blanch et al., Intensive Care Med 2012) and moved to machine learning (Gholami et al., Comput Biol Med 2018; Rehm et al., 2018; Sottile et al., Crit Care Med 2018; Bakkes et al., Comput Methods Programs Biomed 2023, using simulated data), to object-detection models (PVADet, 2025, cross-validation mAP 88% dropping to 66% on the test set) and to 1-D U-Net segmentation of inspiratory/expiratory onsets (Scientific Reports 2026; 9,719 breaths from 33 patients, F1 > 0.98 on asynchronous breaths). A 2025 systematic review (Respiratory Care) found only 13 AI studies with 332 participants in total, despite 5.8 million analysed breaths, and a 2025 real-time circuit-event study collected 3.1 million breaths from 48 patients. The open ICU databases, on the other hand, have no ventilator waveforms at all (MIMIC waveforms carry ECG/ABP/PPG; VitalDB has airway-pressure tracks for some anaesthetised, mostly passive patients), but they chart ventilator and monitor numbers at 2-min (HiRID), 5-min (eICU vitals) or hourly (MIMIC-IV, eICU respiratory charting) resolution for tens of thousands of ventilated stays.

## The research gap

What has been done (2018-2026):

- Waveform detectors are accurate in-sample but are trained on single-centre data of tens of patients with no public benchmark; the PVADet cross-validation-to-test drop is the visible symptom of the generalisation problem.
- The outcome epidemiology of asynchrony rests on small prospective cohorts (Thille 2006: 62 patients; Blanch 2015: 50 patients). It has not been replicated at scale because scale exists only in databases without waveforms.
- Simulated ventilator waveforms have been used for training (Bakkes et al., 2023), but no open, documented generator emits *labelled* asynchrony types together with charted-resolution downsampling.
- The sibling project `ventilation-policy-offline-rl` treats ventilator settings as actions; it has no notion of asynchrony as a state or an outcome.

What is specifically missing (our angle):

1. **A charted-resolution asynchrony surrogate, validated against waveform ground truth.** Ventilators chart the delivered rate; bedside monitors chart an impedance respiratory rate that also counts chest-wall movements from ineffective efforts. The monitor-minus-ventilator rate discrepancy is a footprint of ineffective effort; delivered rate above the set rate is a footprint of double/auto-triggering; breath-to-breath variability of tidal volume and peak pressure is a footprint of breath stacking and flow starvation. Nobody has built, calibrated or validated such a surrogate.
2. **An open simulator with labels** (`lungsim.py`) that produces every major asynchrony type from mechanistic parameters (resistance, compliance, neural rate, effort amplitude, trigger sensitivity, cycling criterion, bias flow), used both to develop detectors and to test surrogates by downsampling to charted resolution.
3. **Cross-dataset detector validation** on whichever open waveform sets are available (checked at project start; see `data/README.md`), with the simulator as the shared development set.
4. **Epidemiology at scale**: prevalence of high surrogate-AI hours and its association with ventilation duration and mortality in HiRID, eICU (208 hospitals) and MIMIC-IV, with confounding control for sedation, mode and severity, and consistency of effect sizes across databases.

## Research questions / hypotheses

1. **RQ1 (detectors).** H1: rule-based detectors tuned on the simulator reach F1 >= 0.85 for ineffective effort and double triggering on open annotated waveform data without re-tuning; a deep model trained on simulator + one dataset transfers to another with < 10-point F1 loss when the simulator covers the second dataset's ventilator settings.
2. **RQ2 (surrogate validity).** H2: at 2-min resolution, the composite trigger-excess index (monitor-minus-ventilator rate plus delivered-minus-set rate) correlates with the true AI at Spearman rho >= 0.6 on waveform data downsampled to charted resolution, and a gradient-boosted mapping from all surrogate features reaches AUROC >= 0.8 for AI >= 10% bins. H2b: the monitor-minus-ventilator rate discrepancy is the strongest single feature for ineffective effort.
3. **RQ3 (prevalence).** H3: >= 20% of ventilated hours in HiRID and MIMIC-IV have surrogate AI >= 10%, more in pressure-support than in controlled modes and more at night.
4. **RQ4 (outcomes).** H4: the fraction of ventilated hours with surrogate AI >= 10% is associated with longer ventilation (cause-specific hazard of extubation < 1) and higher ICU mortality (adjusted OR > 1.2 per 20-percentage-point increase), replicating Blanch et al. (2015) at 100x the sample size.
5. **RQ5 (transportability).** H5: effect sizes are consistent across the three databases (I^2 < 50%) once the surrogate is calibrated per database with the monitor/ventilator pairing observed in that database.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| HiRID v1.1.1 | 2-min charted ventilator variables (set/measured rate, tidal volume, peak pressure, PEEP, minute volume, mode) and monitor respiratory rate; outcomes; pharma (sedation) | ~34k ICU stays | PhysioNet credentialed | https://physionet.org/content/hirid/1.1.1/ |
| eICU-CRD v2.0 | `respiratoryCharting` (ventilator settings/observations), `vitalPeriodic.respiration` (monitor, 5 min), `respiratoryCare`, `infusionDrug`, APACHE, outcomes; 208 hospitals | ~200k stays | PhysioNet credentialed | https://physionet.org/content/eicu-crd/2.0/ |
| MIMIC-IV v3.1 `icu` | `chartevents` ventilator items (RR set 224688, RR total 224690, VT 224685, PIP 224695, PEEP 220339, mode 223849), monitor RR 220210, `procedureevents` ventilation episodes, `inputevents` sedation | ~94k stays | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| Open ventilator-waveform datasets with breath labels | Detector validation and surrogate calibration (candidates listed in `data/README.md`; availability and licence to be verified) | varies | varies (open / on request) | see `data/README.md` |
| Google Brain Ventilator Pressure Prediction | Artificial-lung pressure/flow (no efforts) for segmentation pre-training only | ~125k breaths | Open (Kaggle) | https://www.kaggle.com/c/ventilator-pressure-prediction |
| Simulator (this repo, `lungsim.py`) | Labelled asynchronies from mechanistic parameters; downsampled to charted resolution | any | Generated locally | - |

## Methods

Pipeline (each step maps to a module in `src/pva_detect/`):

1. **Simulation** (`lungsim.py`). Equation of motion Paw = R*Q + V/C + PEEP - Pmus; assist-control volume control or pressure support with backup rate; flow trigger, trigger lockout (minimum expiratory time), bias-flow cap on demand flow, PS cycling at a fraction of peak flow; neural efforts (half-sine Pmus) that are spontaneous or entrained to mandatory breaths; cardiogenic flow oscillations for auto-triggering. Outputs: flow, Paw, volume, Pmus and ground-truth breath/effort tables with labels; scenarios in `SCENARIOS`.
2. **Waveform detectors** (`breaths.py`). Flow-threshold breath segmentation (above bias flow), per-breath Ti, Te, VTi, VTe, PIP, PEEP, instantaneous rate; ineffective effort = expiratory-flow notch of prominence >= 0.04 L/s; double trigger = expiratory time shorter than half the median inspiratory time; asynchrony index; event matching for recall/precision. Deep models (1-D U-Net / object detection) are an optional layer on the same segmentation.
3. **Surrogates** (`surrogates.py`). `bin_breaths` downsamples waveform ground truth to charted bins (delivered rate, monitor rate with a stated ineffective-effort detection probability, VT/PIP/rate variability, minute volume); `surrogate_features` builds the charted-style features; `trigger_excess_index` is the physiologic composite; `fit_surrogate_model` / `evaluate_surrogate` calibrate and evaluate; `charted_window_features` computes identical features from real charted tables (HiRID 2 min, eICU/MIMIC hourly).
4. **Extraction** (`cohort.py`). DuckDB templates for MIMIC-IV chartevents and eICU respiratoryCharting; regex lookups for HiRID/eICU variable names; pivot to wide; ventilator-mode parsing; exposure table per stay; logistic association with cluster-robust SEs (hospital in eICU).
5. **Outcome models.** Cause-specific Cox / Fine-Gray for extubation with death as competing risk (lifelines), logistic for ICU mortality; covariates: age, sex, admission diagnosis group, SOFA/APACHE, PEEP, FiO2, mode class, sedation (RASS where charted; sedative infusion rates), hour of day; hospital random effects in eICU.

### Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/download_data.py --simulate --out data/simulated --duration 600 --seeds 0 1 2
```

```python
from pva_detect import lungsim, breaths, surrogates
r = lungsim.simulate(lungsim.SCENARIOS["ineffective_efforts"])
seg = breaths.segment_breaths(r.flow, r.fs)
f = breaths.breath_features(r.flow, r.paw, r.fs, seg)
ie = breaths.detect_ineffective_efforts(r.flow, r.fs, seg)
print(r.asynchrony_index(), breaths.match_events(ie["idx"], r.efforts.loc[~r.efforts.delivered, "start_idx"], r.fs))
```

## Evaluation & statistics

- Detectors: recall/precision per event type against simulator ground truth (positive control; tests enforce >= 0.7 on both) and against open annotated data with patient-grouped splits; F1 with patient-level bootstrap CIs; false-positive rate per hour on the passive `controlled` scenario (negative control).
- Surrogates: fitted on simulator + waveform data, evaluated on held-out patients/seeds; Spearman rho with true AI, AUROC for AI >= 10% bins, calibration of predicted AI; the ineffective-effort monitor-detection probability is *estimated* on real monitor/ventilator pairs (MIMIC-IV has both 220210 and 224690 at shared chart times) rather than assumed.
- Epidemiology: exposure = fraction of ventilated hours with surrogate AI >= 10% (and mean surrogate AI); adjusted logistic (mortality) and competing-risk (extubation) models; E-values for unmeasured confounding; negative-control outcome (a lab unrelated to ventilation) and negative-control exposure (surrogate computed on non-ventilated hours) to detect residual bias; sedation examined as confounder and mediator (asynchrony leads to more sedation) with a time-updated exposure.
- Multiple comparisons: 2 outcomes x 3 databases x 2 exposure definitions; Benjamini-Hochberg; H1-H5 pre-registered.
- Sensitivity: charting resolution (2 min vs hourly emulated on HiRID), mode class strata, surrogate threshold (AI 10% vs 5%), and the simulator's parameter ranges (R, C, effort amplitude) as a robustness sweep.

## Publishable angle

Headline result: "A charted-data surrogate of patient-ventilator asynchrony, validated against waveform ground truth, shows that asynchrony burden is associated with longer ventilation and higher mortality in 60,000 ventilated stays across three databases and two countries." The simulator and surrogate would let every group with charted ICU data study asynchrony without waveforms.

Target venues: American Journal of Respiratory and Critical Care Medicine or Intensive Care Medicine (epidemiology); Critical Care (surrogate validation); IEEE Transactions on Biomedical Engineering or Journal of Clinical Monitoring and Computing (simulator and detectors).

Follow-ups: add surrogate-AI as a state variable to `ventilation-policy-offline-rl`; prospective validation of the surrogate against a waveform recorder in one ICU; ventilator-brand-specific calibration.

## Risks, confounds & mitigations

- Surrogate validity depends on how each monitor computes respiratory rate (impedance algorithms differ): calibrate per database, report the monitor/ventilator pairing distribution, and validate on waveform data from at least two ventilator brands.
- Charting frequency differs (2 min vs hourly): emulate hourly charting on HiRID to quantify the information loss before pooling; report per-database results separately as well as pooled.
- Mode confounding: pressure support inflates "delivered minus set rate" by design; mode class is a stratification variable, and the surrogate is calibrated per mode class.
- Sedation is both confounder and mediator: time-updated exposure with sedation lagged; mediation analysis as a secondary result.
- Open waveform datasets may not be obtainable: the simulator plus the Kaggle artificial-lung data keep the detector work reproducible; the surrogate can still be validated on any single waveform recording set obtained under agreement.
- The simulator is a single-compartment model without auto-PEEP or leak: state the limits, add an expiratory flow limitation and leak term as a sensitivity extension.

## Milestones

- [ ] Simulator scenarios and detectors validated (tests pass); simulated corpus generated
- [ ] Open waveform datasets obtained and licence recorded; detector transfer evaluated
- [ ] Surrogate calibrated on simulator + waveform data; monitor detection probability estimated on MIMIC-IV pairs
- [ ] PhysioNet credentialing; HiRID/eICU/MIMIC-IV ventilator variables verified and extracted
- [ ] Prevalence tables by database, mode and time of day
- [ ] Outcome models with confounding control, negative controls, E-values
- [ ] Pre-registration (OSF) before outcome analyses; manuscript and simulator release

## Ethics / data-use notes

- HiRID, eICU-CRD and MIMIC-IV are PhysioNet credentialed resources: CITI training, signed DUAs, approved encrypted storage; no redistribution; credentialed data are never sent to third-party LLM APIs or other external services.
- Open waveform datasets are used under their own licences; simulated data carry no privacy risk and are the only data shipped by the code.
- Never commit data; `.gitignore` excludes `data/` and `outputs/`. Publish aggregates only (counts >= 10).
- Credentials are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`; never hard-code them.
