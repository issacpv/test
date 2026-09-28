# QT-Dose: real-world dose-response and time-course of QT-interval prolongation from MIMIC-IV-ECG + eMAR

**One-sentence pitch.** Use the administration-level medication record (MIMIC-IV `emar`/`emar_detail`, ICU `inputevents`) linked to ~800,000 12-lead ECGs (MIMIC-IV-ECG) to estimate, *within patients*, how much each cardio-active drug changes the rate-corrected QT interval (QTc) per unit dose and how the change evolves with time since administration, with a multiverse over heart-rate correction formulas, re-delineated QT intervals, electrolyte adjustment and negative-control drugs/outcomes, and then test whether the real-world slopes recapitulate the published CredibleMeds risk categories and FAERS reporting patterns.

## Status / difficulty / timeline / compute

- Status: design + starter code (this repo). No data are shipped.
- Difficulty: MSc-level (clinical pharmacology + biostatistics + signal processing); PhD-level if the exposure-response hysteresis models and the delineation validation are pushed to full depth.
- Timeline: 6-9 months (1 month credentialing + cohort/exposure tables, 2 months QT measurement + QC, 2 months dose-response modelling and multiverse, 1-2 months external validation vs CredibleMeds/FAERS, remainder writing).
- Compute: CPU only. MIMIC-IV-ECG `machine_measurements.csv` and the MIMIC-IV hosp tables fit in memory on a 32 GB workstation; re-delineating 100k-300k exposure-linked waveforms (WFDB, 500 Hz, 10 s) takes hours on a multi-core CPU. Disk: ~90 GB for the full waveform set, or ~2 GB if only the machine-measured intervals are used.

## Background

Drug-induced lengthening of the rate-corrected QT interval is the leading cause of post-marketing drug withdrawals for cardiac safety, and prolonged QTc is the main surrogate for the pro-arrhythmic risk of torsades de pointes. Regulatory assessment (ICH E14/S7B) has moved decisively toward *concentration-QTc* modelling in dedicated thorough-QT studies: a linear mixed model of the QTc change against plasma concentration is now the primary analysis, with a threshold of interest around a 10 ms mean change (FDA E14/S7B Q&A, 2022; and see the 2025 FDA review of QT reports, Springer J Pharmacokinet Pharmacodyn, doi:10.1007/s10928-025-09985-4). These studies are small (tens of healthy volunteers), single-dose, and by design exclude the sick, poly-medicated, electrolyte-deranged inpatients in whom torsades actually occurs.

The complementary evidence base is (i) curated drug lists such as CredibleMeds/QTdrugs, which classify drugs into known / possible / conditional torsades risk from case reports and trials, and (ii) spontaneous-reporting pharmacovigilance (FAERS), which is prone to reporting bias. What is largely missing is a *dose-resolved, within-patient, time-resolved* estimate of the QTc effect of each drug at the doses actually used in routine inpatient care, learned from the electronic record.

MIMIC-IV now makes this tractable: MIMIC-IV-ECG provides ~800k 12-lead waveforms with machine-measured intervals (including QT and RR) time-stamped to the second, and the MIMIC-IV hosp module provides the barcode medication-administration record (`emar`, `emar_detail`) plus labs (potassium, magnesium, calcium) and demographics; ICU continuous infusions are in `inputevents`. Because many patients have several ECGs before and after a dose, the drug effect can be estimated as a within-patient change, controlling for the patient's own baseline.

## The research gap

**What has been done (2020-2026):**

- **Thorough-QT / concentration-QTc modelling** is the regulatory standard but is confined to small healthy-volunteer studies (ICH E14/S7B Q&A 2022; Garnett et al., 2018, J Clin Pharmacol on C-QTc; FDA insights, 2025, J Pharmacokinet Pharmacodyn doi:10.1007/s10928-025-09985-4). These do not describe routine inpatient doses or drug combinations.
- **AI-ECG risk prediction.** QTNet (Zhang/… et al., 2024, *JACC: Clinical Electrophysiology*, doi:10.1016/j.jacep.2024.01.022) trains a CNN on the ECG plus risk factors to predict *whether* a newly prescribed QT-prolonging drug will induce long-QT in outpatients (AUC ~0.80, external-validated). A 2025 *Heart Rhythm* study (doi:10.1016/j.hrthm.2025... , S1547-5271(25)03026-7) predicts diLQTS risk at prescribing time from the MUSE ECG database. QTcNet/QTcNet-type models (Europace 2025, euaf274) do direct HR-corrected QT measurement. All predict a *binary/threshold* event at prescribing time; none estimate a *per-drug dose-response slope and time-course* from administration-level data.
- **Real-world QT surveillance.** Prior data-driven surveillance (e.g., the 2022 PMC8803188 study using 12-lead and continuous ECG adverse-reaction signals) detects QT signals but does not fit exposure-response curves keyed to eMAR dose.
- **CredibleMeds and FAERS** provide risk categories and disproportionality signals but not dose-resolved effect sizes.

**What is specifically missing (the gap this project fills):**

1. No public study estimates, *per drug, from MIMIC-IV administration data, the within-patient QTc change per unit dose* (a real-world "C-QTc surrogate" using dose as the exposure proxy) with proper baseline control and confounder adjustment.
2. No study characterises the *time-course* (onset/offset, hysteresis between the exposure peak and the QTc peak) of the QTc effect from routinely collected ECGs, exploiting patients with multiple ECGs around a dose.
3. The sensitivity of these estimates to the well-known analyst degrees of freedom - **which QT-correction formula** (Bazett, Fridericia, Framingham, Hodges, or a cohort-fitted individual/population correction), **machine-measured vs re-delineated QT**, **electrolyte adjustment**, and **the exposure window** - has never been mapped as a specification-curve / multiverse in this setting.
4. No head-to-head test of whether *real-world dose-response slopes rank drugs consistently with CredibleMeds categories and FAERS torsades disproportionality*, which would validate the EHR estimand and flag drugs whose real-world signal disagrees with the curated lists.

This is deliberately **not** a torsades-prediction model (QTNet already exists); it is a pharmaco-epidemiologic **estimation** of dose-response and time-course, plus a **methodological audit** of how fragile such estimates are.

## Research questions / hypotheses

1. **RQ1 (dose-response).** For each drug d in a pre-registered QT-drug list, is there a positive within-patient association between administered dose and the change in QTc from the patient's pre-dose baseline ECG to the first post-dose ECG? *H1:* CredibleMeds "known risk" drugs (e.g., sotalol, dofetilide, haloperidol, methadone, azithromycin, ondansetron) show significantly positive dose slopes; "no known risk" negative-control drugs (e.g., aspirin, metoprolol, acetaminophen) show slopes indistinguishable from zero.
2. **RQ2 (time-course / hysteresis).** Does the QTc effect peak at a delay after the administration time consistent with each drug's pharmacokinetics rather than instantaneously? *H2:* fitting QTc change against time-since-dose yields a rise-then-decay curve; the time-to-peak differs across drugs and exceeds zero for orally administered agents.
3. **RQ3 (multiverse robustness).** How much do the estimated per-drug slopes and their significance move across the correction-formula x measurement-source x electrolyte-adjustment x exposure-window grid? *H3:* Bazett inflates rate-dependent slopes for rate-changing drugs (beta-blockers, sotalol) relative to Fridericia/individual correction; drug *rankings* are more stable than absolute slopes.
4. **RQ4 (concordance with external knowledge).** Do real-world dose-response slopes rank-correlate with (a) CredibleMeds risk category and (b) FAERS torsades-de-pointes disproportionality (ROR)? *H4:* Spearman rho > 0.4 with CredibleMeds ordinal risk; drugs with high real-world slope but low FAERS ROR (or vice-versa) are identified as discordant.
5. **RQ5 (effect modification).** Are dose slopes larger in hypokalemia/hypomagnesemia, in females, in the elderly, and under drug-drug combinations (two QT drugs co-administered)? *H5:* slopes increase with lower serum potassium and are larger in females; two-drug co-administration is supra-additive for at least some pairs.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV-ECG v1.0 | 12-lead waveforms + `machine_measurements.csv` (QT, RR, QTc, axis), `record_list.csv` (subject_id, study time) | ~800k ECGs, ~160k patients, 500 Hz, 10 s | Credentialed (PhysioNet CITI + DUA) | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| MIMIC-IV v3.1 hosp | `emar`, `emar_detail` (drug, dose, route, admin time), `pharmacy`, `prescriptions`, `labevents` (K/Mg/Ca), `patients`, `admissions`, `diagnoses_icd` | ~365k patients | Credentialed (PhysioNet) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-IV v3.1 icu | `inputevents` (continuous/bolus infusions, e.g., amiodarone) for exposure in ICU stays | subset | Credentialed (PhysioNet) | https://physionet.org/content/mimiciv/3.1/ |
| CredibleMeds / QTdrugs list | Ground-truth ordinal torsades-risk categories per drug (known / possible / conditional / no risk) | ~200 drugs | Free registration (AZCERT) | https://crediblemeds.org |
| openFDA FAERS | Torsades-de-pointes / long-QT report counts per drug for disproportionality (ROR) | millions of reports | Open API | https://open.fda.gov/apis/drug/event/ |

Notes: MIMIC-IV-ECG intervals are machine-measured (GE/Philips carts); re-delineation is used as a sensitivity analysis and to validate the machine QT. Drug names in `emar` are mapped to RxNorm ingredients before matching to CredibleMeds and FAERS.

## Methods

Pipeline (`src/qt_dose/`):

1. **Exposure construction** (`exposure.py`). Parse `emar`/`emar_detail` administration events; normalise free-text drug strings to RxNorm ingredient (dictionary + fuzzy match); convert doses to a per-ingredient canonical unit (mg; mg/kg using `patients` weight when present); flag route (PO/IV). For ICU infusions, integrate `inputevents` rate x duration into a cumulative exposure. Build, per (patient, drug) episode, the administration timeline.
2. **ECG-exposure linkage** (`linkage.py`). For each administration, find the nearest *pre-dose* ECG (baseline, within a configurable look-back, default 48 h, no intervening dose of the same or another QT drug) and the *post-dose* ECGs within a window (default 0-24 h). Compute time-since-dose for each post-dose ECG. Enforce a wash-out so baselines are drug-naive for that ingredient.
3. **QT measurement** (`qt_measure.py`). Two sources: (a) machine `QT`/`RR`; (b) optional re-delineation from the waveform using a lightweight delineator (`neurokit2` if available; otherwise a template-based fallback shipped here) on lead II / a vector-magnitude lead, with per-beat QT and median-beat QT. Rate correction formulas: Bazett, Fridericia, Framingham, Hodges, and a **cohort-fitted individual/population correction** (regress log QT on log RR in drug-naive baselines to estimate the exponent α, then QTc = QT / RR^α). All implemented in `corrections.py`.
4. **Dose-response model** (`models.py`). Primary estimand: within-patient change ΔQTc = QTc(post) − QTc(baseline). Mixed-effects model per drug: ΔQTc ~ dose + time_since_dose (spline) + serum_K + age + sex + (1 | patient), using `statsmodels` MixedLM; the dose coefficient is the ms-per-unit-dose slope. A time-course model fits ΔQTc against a natural-spline basis in time-since-dose to locate the peak (hysteresis). Robust/cluster SEs; drugs analysed independently with shared code.
5. **Concordance** (`external.py`). Map per-drug slopes to CredibleMeds ordinal risk (Spearman, ordinal logistic) and to FAERS torsades ROR (computed from openFDA counts with a 2x2 disproportionality table and shrinkage). Produce the discordance table.
6. **Multiverse** (`multiverse.py`). Cartesian grid over {correction formula} x {machine vs re-delineated QT} x {electrolyte-adjusted vs not} x {baseline look-back and post-dose window} x {include/exclude rate-changing drugs}; store the slope and CI for every drug in every specification; render a specification curve per drug.

Baselines / comparators: (i) naive between-patient regression of QTc on current dose (to show confounding by indication); (ii) simple pre/post paired t-test per drug (ignores dose and time); (iii) the population-correction mixed model as the main result. Negative-control **drugs** (no-QT-risk medications) and a negative-control **outcome** (PR interval, which should not respond to hERG blockers) bound residual confounding.

Tools: `pandas`, `numpy`, `scipy`, `statsmodels` (MixedLM, splines via `patsy`), `wfdb` (waveform read), optional `neurokit2` (delineation), `requests` (openFDA). No third-party LLM APIs.

## Evaluation & statistics

- **Primary metric.** Per-drug dose slope (ms per canonical dose unit) with 95% CI from the mixed model; time-to-peak (h) from the spline model.
- **Validation scheme.** Within-patient design removes fixed patient confounders; patient is a random effect and never split across "baseline" and "post" in a way that leaks (baseline and post-dose are the same patient by design, which is the point). Sensitivity: leave-one-hospital-service-out, and ICU vs ward.
- **Leakage / confound control.** Exclude baselines with a recent dose of the drug or any other listed QT drug; exclude ECGs during active resuscitation; adjust for concurrent QT drugs; negative-control drugs and the PR-interval negative-control outcome quantify residual bias. Confounding-by-indication is demonstrated explicitly with the naive between-patient model as a foil.
- **Multiple comparisons.** ~40-60 drugs x 5 RQs: Benjamini-Hochberg FDR within each RQ family; report effect sizes with CIs as the primary output, p-values secondary. Specification-curve inference (Simonsohn et al.) with a permutation null across the multiverse.
- **Nulls.** (i) Permutation: shuffle the dose values within a drug's episodes (breaks dose-response, keeps marginal QTc distribution). (ii) Negative-control drugs should yield null slopes; if they do not, the residual bias estimate is subtracted/flagged. (iii) Time-reversal placebo: assign a fake "dose time" before any real dose and check for a spurious ΔQTc.
- **Calibration to regulatory scale.** Report the dose that corresponds to a +10 ms mean QTc change (the ICH E14 threshold) with CI, per drug, to connect to the regulatory estimand.

## Publishable angle

- **Headline.** "Administration-level EHR data recover dose-resolved, time-resolved QTc effects for QT-prolonging drugs that rank-correlate with CredibleMeds risk categories; the estimates are robust to measurement source but sensitive to heart-rate-correction choice for rate-changing drugs; several drugs show real-world/FAERS discordance."
- **Deliverables.** A per-drug real-world dose-response table (slope, time-to-peak, +10 ms dose) with CIs; the specification-curve figures; the RxNorm↔CredibleMeds↔FAERS crosswalk; open analysis code.
- **Target venues.** *Clinical Pharmacology & Therapeutics*; *Journal of the American Medical Informatics Association (JAMIA)*; *Heart Rhythm* or *Europace* (clinical-electrophysiology angle); *Drug Safety* (pharmacovigilance concordance). Methods/benchmark track: CHIL / ML4H.
- **Follow-ups.** Extend the exposure proxy to measured drug levels where available (`labevents` for digoxin, some antibiotics); add MIMIC-IV-ECG re-delineation as a released QT dataset; replicate in eICU / a second credentialed EHR; drug-drug interaction surfaces.

## Risks, confounds & mitigations

- **Confounding by indication** (sicker patients get both QT drugs and abnormal QT). Mitigation: within-patient ΔQTc design, negative-control drugs and outcome, adjustment for acuity/electrolytes; the naive model is shown as the biased comparator.
- **Machine QT measurement error / lead choice.** Mitigation: re-delineation sensitivity analysis; restrict to good signal-quality ECGs; median-beat measurement.
- **Rate-correction artefact.** Rate-changing drugs (beta-blockers, sotalol) confound Bazett. Mitigation: population/individual correction as primary; report all five formulas; PR-interval negative control.
- **Dose harmonisation errors** (free-text drug strings, units, PRN vs scheduled). Mitigation: RxNorm mapping with manual review of the top-frequency strings; unit canonicalisation with audits; exclude ambiguous administrations.
- **Sparse post-dose ECGs / immortal-time issues.** Mitigation: require a genuine pre/post pair; align time at administration; time-varying models; report N per drug and drop drugs with insufficient pairs.
- **Electrolyte timing.** Serum K/Mg may be measured hours from the ECG. Mitigation: carry-forward with a max staleness; sensitivity with/without adjustment (a multiverse axis).

## Milestones

- [ ] PhysioNet credentialing (CITI) + DUA; stage `machine_measurements.csv`, `record_list.csv`, hosp tables via `scripts/download_data.py`.
- [ ] Build RxNorm-normalised exposure table from `emar`/`emar_detail`; freeze the pre-registered QT-drug list + negative controls (from CredibleMeds).
- [ ] ECG-exposure linkage with wash-out; QC of machine intervals; re-delineation on a validation subset.
- [ ] Implement five corrections incl. cohort-fitted α; paired ΔQTc; mixed-model dose slopes per drug.
- [ ] Time-course / hysteresis spline models; time-to-peak per drug.
- [ ] Multiverse grid + specification curves; permutation and negative-control nulls.
- [ ] openFDA FAERS ROR + CredibleMeds concordance; discordance table.
- [ ] Effect modification (K/Mg, sex, age, drug-drug); manuscript.

## Ethics / data-use notes

- MIMIC-IV and MIMIC-IV-ECG are credentialed: complete CITI training and sign the PhysioNet DUA. Store encrypted at rest; **never** send waveforms, notes or derived per-patient tables to third-party LLM/API services except as permitted by PhysioNet's responsible-use policy. Never commit data or derived record-level files.
- Credentials for downloading are read from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD` environment variables only; they are never written to disk or committed.
- CredibleMeds content is used under its terms (research use; cite AZCERT). openFDA is public.
- This is an observational estimation study; it does not make individual prescribing recommendations. Subgroup analyses (sex, age) are reported to expose disparities in risk, not to justify differential care.
