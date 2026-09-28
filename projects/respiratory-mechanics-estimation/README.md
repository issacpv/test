# Respiratory mechanics from charted ventilator time series: state-space estimation of compliance and resistance across HiRID, MIMIC-IV and eICU, and whether driving-pressure risk transports

One-sentence pitch: Treat the sparse, irregular, documentation-dependent ventilator values charted in ICU databases as noisy observations of a latent, time-varying single-compartment lung (elastance E, resistance R), estimate that state with a Kalman/Bayesian filter validated on ground-truth lungs, and test whether the driving-pressure and mechanical-power associations with mortality are invariant across three databases once mechanics are estimated consistently instead of read off inconsistent charts.

## Status / difficulty / timeline / compute

- Status: proposal + starter code (single-compartment forward model and breath-level least-squares fit, Kalman filter/RTS smoother over charted observations with missing values, label mapping and pivoting for MIMIC-IV/eICU/HiRID charting).
- Difficulty: MSc-level for the estimator and validation; PhD-level with the cross-database epidemiology. 6-9 months.
- Compute: CPU only. HiRID (~40 GB raw), MIMIC-IV chartevents (~30 GB), eICU respiratoryCharting (~4 GB); DuckDB/Polars on a 64 GB workstation is sufficient.
- Related projects (kept separate): `ventilation-policy-offline-rl` (uses the same tables as a decision problem), `icu-model-transportability` (cross-database shift), `circadian-label-bias-icu`.

## Background

Lung-protective ventilation targets are defined through respiratory mechanics: driving pressure (Pplat - PEEP) predicts ARDS survival better than tidal volume or PEEP alone (Amato et al., 2015, NEJM); mechanical power integrates pressure, volume and rate (Gattinoni et al., 2016, Intensive Care Med) and is associated with mortality in MIMIC-III and eICU (Serpa Neto et al., 2018, Intensive Care Med); time-varying ventilation intensity tracks daily risk (Urner et al., 2020, Lancet Respir Med); and the benefit of low tidal volume depends on elastance (Goligher et al., 2021, AJRCCM). At the bedside, compliance and resistance come from occlusion manoeuvres or from waveform-based estimators that avoid them (expiratory time constant: Al-Rawas et al., 2013, Crit Care; constrained least squares for spontaneous effort: Vicario et al., 2015, IEEE TBME; model-based PEEP titration: Chiew et al., 2011, BioMed Eng OnLine; inverse modelling overview: Bates, 2009, Cambridge Univ Press). A 2024 lung-model study extended occlusion-free estimation to pressure-controlled ventilation (PMC11107031), and a 2026 Scientific Data release provides daily respiratory mechanics over up to eight days of invasive ventilation for ground truth.

## The research gap

What has been done:

- Retrospective associations between charted driving pressure / mechanical power and outcomes, each restricted to patients with a charted plateau pressure (typically 20-40% of ventilated stays), in MIMIC-III/eICU (Serpa Neto et al., 2018) and in registries (Urner et al., 2020).
- Waveform-based mechanics estimators validated on lung models and small patient series, never applied to the low-resolution charted data that make up the open ICU databases.
- Benchmarks on HiRID (Hyland et al., 2020, Nat Med; Yèche et al., 2021, NeurIPS Datasets & Benchmarks) and MIMIC-IV (Johnson et al., 2023, Sci Data) / eICU (Pollard et al., 2018, Sci Data) use ventilator variables as raw features without a mechanics model.

What is specifically missing:

1. Charted values are documentation artefacts: Pplat is charted only when a nurse or therapist performs a pause (more often in ARDS, in volume-control, at certain hospitals, at certain hours); Ppeak/PEEP/VT/RR are charted far more regularly. Selecting on a charted Pplat induces selection bias that no study has quantified.
2. No estimator exists that combines the physical single-compartment relation (Ppeak - PEEP = E*VT + R*Flow; Pplat - PEEP = E*VT) with the temporal structure of charting (2-min HiRID, ~5-15-min eICU, hourly MIMIC-IV) to produce E and R trajectories with uncertainty for *every* ventilated hour, including hours without a plateau.
3. Transportability of driving-pressure and mechanical-power risk across databases with different charting cadence and case mix has not been examined; differences may be artefacts of measurement rather than biology.

Sharpened angle: an estimator paper (validated on ground-truth lungs and on charted-vs-inferred agreement) plus a transportability paper (does the E/R-outcome relation hold across HiRID, MIMIC-IV, eICU when mechanics are inferred uniformly?).

## Research questions / hypotheses

1. H1 (identifiability): From charted Ppeak, PEEP, VT, RR and inspiratory time alone (no Pplat), a Kalman/RTS state-space model recovers E and R on the Google Brain artificial-lung dataset with known R and C to within 15% median absolute relative error, and on hours where Pplat *is* charted in MIMIC-IV/eICU/HiRID the inferred driving pressure agrees with the charted one (Bland-Altman limits within +/- 4 cmH2O).
2. H2 (selection bias): Stays with a charted Pplat differ systematically (higher PaO2/FiO2 severity, more volume control, specific hospitals in eICU); the driving-pressure hazard ratio for mortality estimated on the Pplat-charted subset differs from that on all ventilated stays using inferred driving pressure (interaction test, p < 0.05).
3. H3 (transportability): After uniform inference, the adjusted association of time-weighted driving pressure with 28-day mortality has overlapping 95% CIs across HiRID, MIMIC-IV and eICU (heterogeneity I^2 < 50%), whereas charted-value associations show I^2 > 50%.
4. H4 (trajectories): Elastance trajectory phenotypes over the first 72 h (latent-class mixed models) are reproducible across databases (adjusted Rand index > 0.6 for cross-fitted assignments) and add prognostic value beyond day-1 values (likelihood-ratio test).
5. H5 (spontaneous effort): In pressure-support / spontaneous modes the single-compartment fit residual increases; a residual-based flag detects mode transitions with AUROC > 0.8, providing a QC criterion.

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| HiRID 1.1.1 | 2-min ventilator observations (Ppeak, PEEP, VT, RR, FiO2, plateau where recorded), outcomes | ~34k ICU admissions (Bern) | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/hirid/ |
| MIMIC-IV 3.x (icu.chartevents, d_items, procedureevents, icustays) | Hourly-ish charted ventilator settings/observations; outcomes | ~94k ICU stays | Credentialed | https://physionet.org/content/mimiciv/ |
| eICU-CRD 2.0 (respiratoryCharting, respiratoryCare, patient, apachePatientResult) | Multi-hospital charted respiratory values | ~200k stays, 208 hospitals | Credentialed | https://physionet.org/content/eicu-crd/ |
| MIMIC-IV demo 2.2 and eICU-CRD demo 2.0.1 | Open samples for pipeline development | 100 patients; ~2.5k stays | Open | https://physionet.org/content/mimic-iv-demo/ , https://physionet.org/content/eicu-crd-demo/ |
| Google Brain - Ventilator Pressure Prediction (Kaggle, 2021) | Artificial-lung breaths with known R and C for estimator validation | ~75k breaths x 80 time steps | Kaggle account + competition rules | https://www.kaggle.com/competitions/ventilator-pressure-prediction |
| Respiratory mechanics & patient-ventilator interaction dataset (Sci Data, 2026) | Daily ground-truth mechanics over up to 8 days of IMV | see paper | Verify host/licence on the paper's data-availability statement | https://www.nature.com/articles/s41597-026-07598-1 |
| AmsterdamUMCdb (optional 4th database) | Replication | ~23k admissions | DUA (free) | https://amsterdammedicaldatascience.nl |

## Methods

1. Cohort: adult ICU stays with invasive mechanical ventilation >= 6 h (episode detection from contiguous ventilator charting, `resp_mech.charting.ventilation_episodes`), first episode per stay; exclude ECMO and tracheostomy-at-admission where identifiable.
2. Variable mapping (`resp_mech.charting.map_labels`): regex over MIMIC-IV `d_items.label`, eICU `respchartvaluelabel`, HiRID `hirid_variable_reference` to canonical names (ppeak, pplat, peep, vt_obs, vt_set, rr, ti, ie, fio2, mode, charted compliance/resistance). Unit harmonisation (mL vs L; L/min vs L/s).
3. Inspiratory flow proxy: VT / Ti with Ti from charted inspiratory time or from RR and I:E; documented as an approximation for decelerating-flow modes (sensitivity analysis by mode).
4. Estimator (`resp_mech.state_space.KalmanMechanics`): state [E, R] random walk with process noise scaled by elapsed time; observations Ppeak - PEEP = E*VT + R*Flow (every charted row) and Pplat - PEEP = E*VT (when charted); RTS smoothing; posterior SD reported. Point estimates from the algebraic single-compartment relations (`resp_mech.single_compartment`) as baseline.
5. Ground truth: Kaggle artificial lung (breath-level least squares and filter on subsampled "charted" summaries of each breath); Sci Data 2026 daily mechanics; charted-vs-inferred agreement.
6. Outcome models: Cox / logistic regression of 28-day mortality on time-weighted driving pressure, mechanical power (Gattinoni formula), E and R, adjusted for age, sex, SOFA/APACHE, PaO2/FiO2, PEEP, VT/PBW, mode; database-stratified and pooled with random effects; latent-class trajectories (`lcmm`-style, or scikit-learn GMM on spline coefficients) for H4.
7. Tools: DuckDB/Polars for extraction, pandas, numpy/scipy, statsmodels, lifelines, scikit-learn.

## Evaluation & statistics

- Estimator validation: median absolute relative error and 90th percentile for E and R; Bland-Altman for driving pressure; coverage of 95% posterior intervals.
- Selection bias (H2): inverse-probability weighting for "Pplat charted" with hospital, mode, severity; compare weighted vs unweighted associations.
- Transportability (H3): per-database hazard ratios with 95% CI; Cochran's Q and I^2; leave-one-database-out prediction of mortality risk (calibration slope, Brier).
- Multiple comparisons: five pre-registered hypotheses; secondary exposure-outcome pairs Holm-corrected.
- Leakage: outcome models never use post-outcome charting; trajectory features restricted to the first 72 h; hospital-level cross-validation in eICU.
- Nulls: permute E/R across stays within database to obtain null hazard ratios; synthetic charting simulator (from the forward model) to check that the estimator does not manufacture associations.

## Publishable angle

Headline: "Inferring rather than charting respiratory mechanics removes selection bias and reveals that driving-pressure risk is stable across three ICU databases" (or the equally useful finding that it is not, pointing to case-mix rather than documentation).

Target venues: Intensive Care Medicine or Critical Care (clinical); IEEE Transactions on Biomedical Engineering or Journal of Clinical Monitoring and Computing (estimator); Scientific Data (if a harmonised mechanics table for the three databases is released as a derived dataset under the PhysioNet credentialed licence).

Follow-ups: feed inferred mechanics into `ventilation-policy-offline-rl` as state variables; extend to two-compartment/viscoelastic models where 2-min HiRID resolution allows; patient-ventilator asynchrony proxies from residual structure.

## Risks, confounds & mitigations

- Charted "Ppeak" in pressure-control modes equals the set inspiratory pressure, and flow is decelerating: model mode explicitly; primary analysis in volume-control, sensitivity in PC/PRVC.
- Spontaneous effort violates the passive single-compartment model: residual-based flags (H5), restrict to controlled modes and sedated periods where documented.
- Auto-PEEP: use total PEEP where charted; otherwise report as limitation.
- Unit and label heterogeneity across 208 eICU hospitals: regex mapping audited manually; per-hospital distribution checks; hospitals with unusable charting excluded with pre-specified rules.
- Time-stamp granularity (MIMIC-IV hourly): the filter's process noise adapts to gaps; uncertainty widens accordingly and is reported.
- Ground-truth lung data are artificial: complement with the 2026 Sci Data patient dataset.

## Milestones

- [ ] Download demo databases and the Kaggle lung dataset; validate estimator on artificial lung (H1a).
- [ ] Obtain credentialed access; extract ventilator charting from MIMIC-IV, eICU, HiRID into a harmonised long table.
- [ ] Ventilation episodes, mode classification, flow proxy; charted-vs-inferred agreement (H1b).
- [ ] Selection-bias analysis of Pplat charting (H2).
- [ ] Outcome models and cross-database heterogeneity (H3); trajectories (H4); QC flags (H5).
- [ ] Manuscripts; release code and the derived-variable specification.

## Ethics / data-use notes

- HiRID, MIMIC-IV and eICU are credentialed PhysioNet resources (CITI training + DUA); store on approved systems, never commit data, never send records to third-party LLM/API services except per PhysioNet's responsible-use policy. Demo databases are open but still governed by their licences.
- Derived tables (per-stay mechanics) remain patient-level and are shared only via PhysioNet as a credentialed derived dataset if at all.
- Report results by sex and age; eICU hospital identifiers are used only for clustering, not for ranking hospitals.
