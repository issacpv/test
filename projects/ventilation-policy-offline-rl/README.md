# ventilation-policy-offline-rl

**Can we trust an off-policy estimate of a ventilator policy?** An off-policy-evaluation (OPE) reliability
audit for offline RL on mechanical ventilation (tidal volume, PEEP, FiO2, weaning) across MIMIC-IV v3.1,
eICU-CRD and HiRID, using cross-site behaviour policies as quasi-ground-truth, a calibrated synthetic MDP
with exact values, and guideline-constrained conservative policies (CQL/IQL with ARDSNet hard constraints).

## Status / difficulty / timeline / compute

- Status: design + starter code. No results yet.
- Difficulty: PhD-level (methods) / strong MSc (benchmark only). 9–12 months.
- Compute: a workstation with 64 GB RAM; DuckDB over the MIMIC-IV / eICU CSV/Parquet files (~10 GB
  compressed) needs no database server. Tabular / linear OPE and CQL-lite run on CPU in minutes; deep
  CQL/IQL (PyTorch, optional) on a single GPU in hours.
- Data access: PhysioNet credentialing (CITI "Data or Specimens Only Research" + signed DUA) for all
  three ICU databases; ~2–4 weeks lead time.

## Background

Offline RL for ICU treatment became visible with the AI Clinician for sepsis (Komorowski et al., 2018,
*Nat Med*) and was quickly criticised for how it was *evaluated*: importance-sampling estimates with
effective sample sizes of a few dozen, sensitivity to the behaviour-policy model, and confounding
(Gottesman et al., 2018, arXiv "Evaluating reinforcement learning algorithms in observational health
settings"; Gottesman et al., 2019, *Nat Med* "Guidelines for reinforcement learning in healthcare";
Jeter et al., 2019, arXiv). Mechanical ventilation followed the same path: VentAI (Peine et al., 2021,
*npj Digit Med*; MIMIC-III + eICU, 7³ action grid), DeepVent (Kondrup et al., 2023, *AAAI*; CQL on
MIMIC-IV), a BCQ policy trained on eICU and tested on MIMIC-IV with OPE (J Med Internet Res 2024;
26:e44494), guideline-informed RL (den Hengst et al., 2024, *Artif Intell Med*), IntelliLung (arXiv
2506.14375, 2025; six ventilator settings, hybrid action spaces, MIMIC-IV/eICU/HiRID), a kNN-simulator
benchmark of seven offline RL algorithms on MIMIC-IV + eICU (Chang & Kuo, 2026, *AAAI*), and digital-twin
verification (arXiv 2603.11372, 2026). All of these *report* OPE values (WIS, FQE, or simulator returns)
as evidence that the learned policy beats clinicians; none of them can say whether those numbers are
right, because there is no ground truth in observational data.

Outside healthcare, OPE benchmarks with ground truth exist (Voloshin et al., 2021, *NeurIPS D&B*; Fu et
al., 2021, *ICLR* "DOPE"); inside healthcare, model-selection studies (Tang & Wiens, 2021, *MLHC*) and
tabular sepsis MDPs (Oberst & Sontag, 2019, *ICML*; ICU-Sepsis, Choudhary et al., 2024, *RLC*) provide
partial answers. Nobody has built an OPE-reliability audit for *ventilation* with (i) cross-site
quasi-ground-truth and (ii) explicit guideline constraints.

## The research gap

**What is missing.**

1. *Cross-site quasi-ground-truth.* The clinician policy of site B (eICU or HiRID) is an executed policy
   whose realised outcomes we observe. Fit a behaviour-cloning model of B's policy, treat it as the
   *target* policy, and estimate its value with OPE (WIS, PDIS, DR, FQE) *using site A's (MIMIC-IV)
   data*. Compare against B's observed, covariate-transported outcome rate. This gives each OPE
   estimator a bias/RMSE and a CI-coverage number on real ICU data — something the published
   ventilation-RL papers cannot report. Repeat for all ordered site pairs (6 pairs) and for policy
   perturbations (e.g. "site B policy with 10 % lower tidal volumes").
2. *Calibrated synthetic ground truth.* A tabular MDP whose state clusters, transition kernels and
   behaviour policy are fit to MIMIC-IV (as ICU-Sepsis did for sepsis), where exact policy values are
   solvable, to test estimator behaviour under controlled overlap violation, horizon and reward design.
3. *Guideline-constrained conservative policies.* CQL / IQL with ARDSNet lung-protective ventilation
   as hard constraints (VT 4–8 mL/kg PBW, plateau ≤ 30 cmH2O, PEEP/FiO2 tables, SpO2 88–95 %)
   and a clinician-editable constraint set; evaluate only with the estimators the audit shows to be
   reliable, and quantify the price of constraints in estimated value and in support (effective sample
   size).
4. *External validation as first-class output*: report every value estimate with its ESS, its
   bootstrap CI, and its cross-site disagreement, and pre-specify the reliability criteria an estimate
   must pass before a claim "better than clinicians" is made.

If a 2026 paper publishes a ventilation OPE benchmark with ground truth, sharpen towards the
cross-site design (1) and the guideline-constraint price (3), which remain unaddressed.

## Research questions / hypotheses

1. **H1.** On the cross-site quasi-ground-truth task, WIS and PDIS have |bias| > 5 percentage points
   of survival and CI coverage < 60 % once the target policy differs from the behaviour policy on
   > 30 % of state-action pairs; DR and FQE are less biased but FQE's CI is over-confident.
2. **H2.** Estimator rank agreement (Spearman ρ between OPE ranking and ground-truth ranking of 20
   candidate policies) exceeds 0.7 only for estimators with ESS > 200 per 10,000 trajectories.
3. **H3.** Value estimates of previously published unconstrained CQL-style policies drop by ≥ 50 % of
   their claimed improvement when re-evaluated with a behaviour policy learned on a held-out site
   (sensitivity to behaviour-policy misspecification).
4. **H4.** Adding ARDSNet hard constraints to CQL reduces the estimated improvement over clinicians
   by < 20 % of its value while raising ESS by > 2× (constrained policies stay in support).
5. **H5.** In the calibrated synthetic MDP, the estimator error ordering learned in simulation predicts
   the cross-site error ordering on real data (concordance of rankings across the 6 site pairs).

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (hosp + icu) | `icustays`, `chartevents` ventilator settings (via mimic-code derived `ventilator_setting`, `ventilation`, `oxygen_delivery`, `bg`, `vitalsign`, `sofa`), `admissions` (mortality), `patients` (age/sex), `omr`/height for PBW | ~65k ICU stays; ~20k ventilated | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV Demo v2.2 | 100 patients, same schema; used for `--sample` pipeline tests | small | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| eICU-CRD v2.0 | `respiratoryCharting` (Tidal Volume (set), PEEP, FiO2, Plateau Pressure), `respiratoryCare`, `vitalPeriodic`, `lab`, `patient` (hospital mortality, unit type, hospital id → 208 hospitals) | 200k ICU stays; ~30k ventilated | Credentialed (PhysioNet) | https://physionet.org/content/eicu-crd/2.0/ |
| eICU-CRD Demo v2.0.1 | subset for `--sample` | small | Open | https://physionet.org/content/eicu-crd-demo/2.0.1/ |
| HiRID v1.1.1 | high-resolution (2-min) ventilator and gas-exchange variables; `observation_tables`, `pharma`, `general_table` | 34k ICU stays (Bern) | Credentialed (PhysioNet) | https://physionet.org/content/hirid/1.1.1/ |
| ARDSNet protocol tables | PEEP/FiO2 tables, VT/plateau/pH/oxygenation targets (NEJM 2000; Brower et al., 2004, NEJM) | — | Open | https://www.ardsnet.org/ (protocol card) |
| Sepsis simulator / ICU-Sepsis MDP | external tabular MDPs to cross-check the synthetic harness | — | Open (GitHub) | Oberst & Sontag 2019; Choudhary et al. 2024 |

## Methods

1. **Cohort & MDP construction** (`vent_rl.mdp_builder`, DuckDB SQL)
   - Adults, invasive ventilation ≥ 24 h, first ICU stay; exclude tracheostomy on admission and
     comfort-care. Time from intubation to extubation/death/day 7, in 4-h bins (sensitivity: 1-h and 8-h).
   - State (per bin): FiO2, PEEP, VT/kg PBW, plateau, driving pressure, RR, minute ventilation, SpO2,
     PaO2, PaCO2, pH, HR, MAP, temperature, lactate, SOFA sub-scores, vasopressor dose, sedation (RASS),
     fluid balance, age, sex, weight, hours ventilated. Missingness indicators + forward fill ≤ 8 h.
   - Actions: discretised (VT/kg PBW × PEEP × FiO2) grid, default 3×3×3 = 27 (clinical bins: VT
     < 6 / 6–8 / > 8; PEEP < 8 / 8–12 / > 12; FiO2 < 0.4 / 0.4–0.6 / > 0.6), optional 7³ as in VentAI,
     plus a "wean/SBT" action.
   - Reward: terminal +1 survival / −1 death (hospital, or 90-day where available), optional shaped
     intermediate rewards (SpO2 / driving-pressure targets); reward design is an audited axis, not fixed.
   - States clustered with k-means (k = 500–750) for tabular estimators; raw features for FQE/CQL.
2. **Behaviour policy models** (`vent_rl.policies.behavior_cloning`): tabular counts with Dirichlet
   smoothing and a multinomial logistic model; report cross-entropy and calibration per site.
3. **OPE estimators** (`vent_rl.ope`): WIS, PDIS, doubly-robust (DR with FQE Q-model), FQE-lite
   (linear / tabular); bootstrap CIs; ESS; overlap diagnostics.
4. **Policies** (`vent_rl.policies`): behaviour cloning, tabular CQL-lite (conservative penalty on
   out-of-data actions), optional PyTorch CQL/IQL; `vent_rl.constraints` masks disallowed actions
   given the current state (ARDSNet tables) before argmax.
5. **Reliability harness** (`vent_rl.reliability`): synthetic tabular MDP with exact value via linear
   solve; site-shift generator (perturb transitions and behaviour policy); bias/RMSE/coverage/rank
   agreement per estimator; cross-site protocol on real data (train BC on site B → target; OPE on site
   A; ground truth = transported outcome on B via inverse-probability-of-site weighting on baseline
   covariates).
6. **Baselines**: clinician policy (BC), random-in-support policy, published DeepVent-style CQL
   without constraints, "always ARDSNet" rule policy.

Libraries: `duckdb`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`; optional `torch`,
`d3rlpy` (CQL/IQL/FQE reference implementations).

## Evaluation & statistics

- Ground-truth quantities: exact value (synthetic); transported observed survival on site B (real).
- Estimator metrics: bias, RMSE, 95 % CI coverage (bootstrap over trajectories, 1,000 draws),
  ESS, Spearman rank agreement across candidate policies; all stratified by policy divergence
  (mean total-variation distance from behaviour).
- Validation scheme: no cross-site leakage — state clustering and behaviour models are fit per site on
  training splits; the held-out site is never used for tuning. Patients split by `subject_id` /
  `patientunitstayid` / `patientid`; eICU also split by hospital to test within-site transport.
- Multiple comparisons: results are estimator × site-pair × divergence-bin cells; report all cells
  with Holm-adjusted tests only for the pre-specified H1–H5.
- Nulls: a target policy equal to the behaviour policy must return the observed mean (sanity),
  and a random permutation of actions must give ESS-consistent variance.
- Clinical plausibility: guideline compliance rate of each policy (`constraints.compliance_rate`),
  fraction of recommendations outside the observed clinician range per state cluster.

## Publishable angle

Headline: "On real cross-site ICU data, importance-sampling OPE of ventilator policies is wrong by X
points of survival with Y % CI coverage; FQE/DR are usable only when ESS > Z; guideline-constrained
CQL keeps 80 % of the estimated gain at 2× the support" — the first ground-truth-anchored reliability
statement for the ventilation-RL literature, with a reusable audit harness.

Venues: *npj Digital Medicine*, *Lancet Digital Health* (if clinical framing dominates), *Machine
Learning for Healthcare (MLHC)*, *CHIL*, *NeurIPS Datasets & Benchmarks*; *Critical Care Medicine*
(clinical companion).

Follow-ups: extend the audit to sepsis (AI Clinician replication across sites), to continuous-time
estimators, and to clinician-in-the-loop constraint elicitation studies.

## Risks, confounds & mitigations

| Risk | Mitigation |
|---|---|
| "Quasi-ground-truth" on site B is itself confounded by case-mix | Transport with IPW on baseline covariates; report sensitivity (E-values) and a synthetic check with known shift |
| Variable definitions differ across MIMIC/eICU/HiRID (set vs observed VT; PEEP total vs set) | Harmonised concept table with explicit provenance; audited by an intensivist; sensitivity to definitions |
| Height missing → PBW uncertain | Use `omr` height, then sex-specific median imputation with an indicator; sensitivity excluding imputed |
| Behaviour-policy misspecification dominates IS estimators | Two behaviour models (tabular / logistic) and clipping; report both |
| Reward hacking via intermediate rewards | Terminal-only main analysis; shaped rewards as an audited axis |
| Small ESS makes CIs meaningless | Pre-specified ESS floor (200) below which no value claim is made |
| Actions recorded at charting, not decision, times | 4-h bins with last-observation semantics; 1-h sensitivity with HiRID |
| PhysioNet data must not leave approved environments | See ethics section; no LLM APIs on the data |

## Milestones

- [ ] PhysioNet credentialing; download MIMIC-IV v3.1, eICU-CRD, HiRID (`scripts/download_data.py`).
- [ ] Concept harmonisation table; DuckDB cohort SQL for the three sites; `--sample` run on the demo datasets.
- [ ] MDP tables (4-h bins, 27 actions, terminal reward); descriptive comparison of behaviour policies across sites.
- [ ] Synthetic calibrated MDP + estimator audit (bias / coverage / rank agreement).
- [ ] Cross-site quasi-ground-truth audit for 6 site pairs and 20 perturbed policies.
- [ ] CQL-lite / IQL with and without ARDSNet constraints; support and compliance metrics.
- [ ] Pre-registered reliability criteria; manuscript; release harness and SQL.

## Ethics / data-use notes

- MIMIC-IV, eICU-CRD and HiRID are credentialed PhysioNet resources: complete CITI training, sign
  the DUA, and keep the data on approved machines. PhysioNet's responsible-use policy forbids sending
  the data to third-party LLM APIs (except the specific pathways PhysioNet lists); do not paste rows
  into chat tools. Never commit data (`data/` is git-ignored); credentials come from environment
  variables (`PHYSIONET_USER`, `PHYSIONET_PASSWORD`).
- Outputs are retrospective decision-support research, not a device; no policy is to be used for
  patient care. Report subgroup performance (sex, age, race where recorded) for any learned policy.
- eICU hospital identifiers are used only as grouping factors, not reported individually.
