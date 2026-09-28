# Slow-oscillation-spindle coupling as a cross-cohort aging biomarker: normative curves, transportability and cardiovascular outcomes

**One-sentence pitch.** Build one harmonized YASA-based spindle/slow-oscillation (SO) coupling pipeline across Sleep-EDF, SHHS, MESA, MrOS, CFS and open hd-EEG sleep datasets, fit lifespan normative (centile) curves for coupling phase precision by age and sex, and test whether coupling precision is a better and more *transportable* predictor of age, cognition and incident cardiovascular disease than the classic spindle-density metrics.

## Status / difficulty / timeline / compute

- Status: design + starter code; no data committed.
- Difficulty: MSc-to-PhD; 6-9 months for the normative + transportability paper (Sleep-EDF, SHHS, MESA, MrOS, CFS), +3-6 months for the cardiovascular-outcome arm.
- Compute: CPU only. YASA processes one whole-night EDF in ~1-2 min per channel; ~12,000 PSGs take ~1-2 days on a 32-core node. Storage: SHHS1+2 ~ 250 GB, MESA ~ 150 GB, MrOS ~ 200 GB, CFS ~ 50 GB, Sleep-EDF ~ 8 GB (stream-and-delete pipelines can keep the footprint under 50 GB).

## Background

Sleep spindles (11-16 Hz, thalamocortical) and cortical slow oscillations (0.3-1.5 Hz) are coupled: spindles preferentially occur near the SO up-state, and the *precision* of this timing predicts overnight memory consolidation (Helfrich et al., 2018, *Neuron*; Muehlroth et al., 2019, *Sci. Rep.*; Hahn et al., 2020, *eLife*). Coupling precision decays with age and with prefrontal atrophy (Helfrich et al., 2018), the coupling hierarchy reverses around midlife (Kurz et al., 2023, *J. Neurosci.*), and mistimed coupling tracks early amyloid burden (Chylinski et al., 2022, *eLife*). Spindle density, in contrast, has robust lifespan norms from 11,630 NSRR participants (Purcell et al., 2017, *Nat. Commun.*) and was recently linked to incident coronary and cardiovascular disease in SHHS (Juginovic et al., 2025, *Sleep Med.*).

Coupling metrics are therefore mechanistically attractive but have been studied almost exclusively in small lab cohorts (n = 20-100) with high-density EEG. Whether they scale to the two-channel, 125-256 Hz clinical PSG of epidemiological cohorts, how they vary with age and sex across the lifespan, and whether models built on them transport across cohorts, montages and devices, is unknown.

## The research gap

**What has been done.**

- Normative spindle metrics: Purcell et al. (2017) fitted age trajectories for spindle density, amplitude, duration and frequency across NSRR cohorts (4-97 y) but did not analyze SO-spindle coupling; a normative spindle database from 772 healthy adults (bioRxiv, 2022) likewise omits coupling.
- Coupling in epidemiological data: Djonlagic et al. (2021, *Nat. Hum. Behav.*) analyzed 3,819 older adults (MESA + MrOS) and found spindle/SO morphology and coupling among the 23 metrics most predictive of processing speed, with marked age-related reductions in coupling magnitude; they did not build centile curves, include young/middle-aged cohorts, or test prediction-model transport. Adra et al. (2022, *Sleep*) tuned spindle-detection parameters for cognition prediction in MrOS.
- Aging and coupling mechanism: Helfrich et al. (2018), Kurz et al. (2023), "Spindle-slow wave coupling and problem-solving skills: impact of aging" (*Sleep*, 2024), and a Bayesian meta-analysis of SO-spindle coupling and memory (2025) all use n < 150 samples. "Phase precession of spindle-slow oscillation coupling across the human brain" (bioRxiv, 2025; 101 participants, 20-85 y) reports age-related decline of coupling precession with no sex difference, again in one lab.
- Brain age: sleep-EEG brain-age models (Sun et al., 2019, *Neurobiol. Aging*; Paixao et al., 2020, *Neurobiol. Aging*; an 18,000-PSG model in 2024) show excess brain age predicts mortality, but they are black-box spectral models, not interpretable coupling metrics.
- Cardiovascular outcomes: Juginovic et al. (2025) linked spindle density (inverse) and ORP-defined sleep depth to CHD/CVD incidence and mortality in SHHS1 (n = 5,782); "Association of disrupted delta wave activity during sleep with long-term cardiovascular disease and mortality" (*JACC*, 2024) used SHHS delta power. Coupling has not been tested for CVD outcomes, and no MESA replication exists.
- Sex differences: a 2026 systematic review/meta-analysis of biological sex effects on spindles and slow waves (bioRxiv) finds the evidence inconsistent and underpowered; large-N sex-stratified coupling norms do not exist.
- Harmonization: the SHARE initiative ("Why harmonizing cohorts in sleep is a good idea and the labor of doing so", 2024) documents the practical difficulty of pooling NSRR cohorts; Luna (Purcell) and YASA (Vallat & Walker, 2021, *eLife*) are the open detectors used at scale.

**What is specifically missing.**

1. **Lifespan normative charts for SO-spindle coupling** (coupling phase precision = mean resultant length, preferred phase, coupled-spindle fraction, event-locked sigma power) pooled across >= 5 cohorts spanning 6-101 y, with sex-specific centiles, in the style of brain-chart normative modeling (Bethlehem et al., 2022, *Nature*; GAMLSS, Rigby & Stasinopoulos, 2005).
2. **A head-to-head transportability test**: age and cognition predictors built from coupling metrics vs spindle-density metrics, evaluated leave-one-cohort-out across cohorts that differ in montage (Fpz-Cz, C4-M1, Fz-Cz, C3-M2), sampling rate (100-256 Hz), device generation (1990s Compumedics to 2010s Embla), and age range; with and without site harmonization.
3. **Coupling vs density for incident cardiovascular disease** in SHHS (discovery) and MESA (replication), adjusting for AHI, sleep depth and spindle density, to test whether thalamocortical timing carries cardiovascular information beyond spindle count.
4. **Sex-stratified coupling trajectories** with adequate power (thousands per sex) and an explicit sex-by-age interaction test.

## Research questions / hypotheses

1. **H1 (normative trajectory).** Coupling phase precision (MRL of SO phase at spindle peak, C4-M1/C3-M2 or nearest available derivation) declines monotonically after ~40 y with a steeper slope than spindle density (standardized slope ratio > 1.5) and rises through childhood/adolescence (CFS).
2. **H2 (sex).** Women show higher coupling precision than men at matched age (Cohen's d ~ 0.2), with no sex x age interaction, whereas spindle density shows a larger sex difference (d > 0.3).
3. **H3 (age transportability).** In leave-one-cohort-out prediction of chronological age, a coupling-feature model loses less accuracy across cohorts (increase in MAE < 20%) than a density-feature model (> 35%), because coupling phase is invariant to amplitude scaling and montage gain whereas density is sensitive to detection thresholds and montage.
4. **H4 (cognition).** Coupling precision predicts processing speed (MESA: Digit Symbol Coding; MrOS: Trails B, 3MS) beyond spindle density and beyond age, sex, AHI and total sleep time, and the association replicates across MESA and MrOS with harmonized effect sizes overlapping.
5. **H5 (cardiovascular outcomes).** In SHHS1, lower coupling precision predicts incident CVD (HR per SD ~ 1.15) independent of spindle density, AHI and ORP-like depth; the association replicates in MESA incident events.
6. **H6 (device/montage invariance).** Within-subject coupling precision measured on Fpz-Cz vs Pz-Oz (Sleep-EDF) and on Fz-Cz vs C4-M1 (MESA) correlates > 0.7, whereas spindle density correlates < 0.5 across derivations.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| Sleep-EDF Expanded (PhysioNet sleep-edfx 1.0.0) | SC: 153 nights, 78 healthy subjects 25-101 y; ST: 44 nights; Fpz-Cz + Pz-Oz at 100 Hz, R&K hypnograms | 197 PSGs, ~8 GB | Open (PhysioNet, no login) | https://physionet.org/content/sleep-edfx/1.0.0/ |
| SHHS (NSRR) | SHHS1 5,793 PSGs (1995-98, 39-90 y) + SHHS2 2,651; EEG C4-A1/C3-A2 at 125 Hz; adjudicated CVD outcomes (`shhs-cvd`) | ~250 GB | Free registration + NSRR data-access request (DUA) | https://sleepdata.org/datasets/shhs |
| MESA Sleep (NSRR) | 2,056 PSGs (2010-13, 54-95 y, six ethnic groups); EEG Fz-Cz, Cz-Oz, C4-M1 at 256 Hz; cognition (MESA exam 5/6) and incident CVD events | ~150 GB | DUA (NSRR); some outcome files require separate MESA approval | https://sleepdata.org/datasets/mesa |
| MrOS Sleep (NSRR) | Visit 1: 2,911 men >= 65 y (2003-05); Visit 2: 1,026; EEG C3-A2/C4-A1 at 256 Hz; 3MS, Trails B | ~200 GB | DUA (NSRR) | https://sleepdata.org/datasets/mros |
| CFS (NSRR) | Visit 5: 730 PSGs, 6-88 y, family-based; EEG C3/C4 at 128 Hz | ~50 GB | DUA (NSRR) | https://sleepdata.org/datasets/cfs |
| HMC Sleep Staging (PhysioNet) | 151 clinical PSGs, F4-M1/C4-M1/O2-M1/C3-M2 at 256 Hz | ~15 GB | Open (PhysioNet) | https://physionet.org/content/hmc-sleep-staging/ |
| Dreem Open Datasets DOD-H / DOD-O | 25 healthy + 55 OSA PSGs, 12 channels at 250 Hz, 5 scorers | ~20 GB | Open (public S3) | Guillot et al., 2020, *IEEE TNSRE* |
| OpenNeuro sleep EEG (hd-EEG) | high-density overnight EEG in BIDS for the montage-invariance test (H6) | varies | Open (openneuro-py / datalad) | https://openneuro.org (search "sleep") |
| NSRR harmonized covariates | `nsrr_age`, `nsrr_sex`, `nsrr_bmi`, `nsrr_ahi_hp3u`, `nsrr_ttldursp_f1` etc. | small | with each NSRR dataset | https://sleepdata.org/datasets/*/variables |

## Methods

1. **Ingest and harmonize (`spindle_age.io_edf`).** Per-cohort channel maps resolve a canonical derivation (priority C4-M1 > C3-M2 > Fz-Cz > Fpz-Cz), re-referencing from referential C4/M1 when needed (MrOS, CFS). Signals are band-passed 0.3-35 Hz and resampled to 100 Hz. Hypnograms: NSRR XML (profusion) parser; Sleep-EDF EDF+ annotations via MNE. Stages coded W=0, N1=1, N2=2, N3=3, R=4.
2. **Event detection (`spindle_age.detect`).** YASA `spindles_detect` (N2+N3, 12-15 Hz fast / 9-12 Hz slow spindles, default relative-power, correlation and RMS thresholds) and `sw_detect` (0.3-1.5 Hz, amplitude criteria); a dependency-free fallback detector is included for tests and for QC of detector sensitivity. Artifact epochs flagged with YASA `art_detect`; nights with < 3 h NREM or > 30% artifact are excluded.
3. **Coupling (`spindle_age.coupling`).** SO phase from the Hilbert transform of the 0.3-1.5 Hz signal; each spindle whose peak falls within +/-1.2 s of an SO negative peak is "coupled"; metrics per night: coupled fraction, preferred phase (circular mean), coupling precision (mean resultant length, MRL), Rayleigh z, Tort modulation index between SO phase and sigma amplitude over all NREM, event-locked sigma power around SO troughs, and surrogate-normalized MRL (z-score against random-position surrogates within SO windows, which removes the dependence of MRL on the number of coupled events).
4. **Normative curves (`spindle_age.normative`).** Quantile regression with cubic B-spline age bases (df = 5) at the 5th/25th/50th/75th/95th centiles, fitted per sex and pooled, with a GAMLSS (BCCG) sensitivity analysis in R via `rpy2` or `gamlss`. Each participant receives a centile and z-score; cohort is entered as a random effect in the mixed-model comparison.
5. **Transportability (`spindle_age.transport`).** Leave-one-cohort-out (LOCO) prediction of age and of cognition with ridge/gradient-boosting on (a) coupling features, (b) density features, (c) both; site harmonization by a ComBat-style location/scale adjustment residualized on age and sex; calibration slope and MAE per held-out cohort; paired subject-level bootstrap of absolute-error differences between feature sets.
6. **Outcomes.** Cox models for incident CVD/CHD in SHHS1 (follow-up from `shhs-cvd-events`) with coupling MRL z-score, spindle density, AHI, age, sex, BMI, hypertension, diabetes, smoking; replication in MESA incident events; Fine-Gray competing risk for non-CVD death.

## Evaluation & statistics

- Pre-registered hypotheses H1-H6; Holm correction within each family.
- Normative fit: cross-validated pinball loss per quantile; centile calibration (fraction below each centile in held-out cohorts); age-slope contrasts via bootstrap of standardized slopes (2,000 resamples, subject-level, family-clustered in CFS).
- Transportability: LOCO MAE and R-squared vs within-cohort 5-fold reference; transport gap = LOCO MAE - within MAE; feature-set differences by paired bootstrap with 95% percentile CI; calibration slope with CI.
- Cognition/CVD: mixed models with cohort random intercept; Cox proportional-hazards with Schoenfeld checks; effect sizes reported per SD of the harmonized metric.
- Leakage prevention: harmonization parameters, detector thresholds and model hyperparameters are estimated on training cohorts only and applied to the held-out cohort; repeated nights from the same subject (SHHS1/2, MrOS V1/V2, Sleep-EDF two nights) are grouped so no subject is split across folds.
- Nulls: surrogate coupling (random positions within SO windows) for MRL; time-shifted spindles for MI; label-permutation null for cognition models.
- Detector robustness: repeat with Luna (Purcell) spindle/SO detectors and with the fallback detector; report agreement (ICC) per cohort.

## Publishable angle

- **Headline.** "Lifespan normative charts of SO-spindle coupling precision from ~12,000 nights across six cohorts: coupling precision declines faster than spindle density, transports across cohorts and montages with half the accuracy loss, and predicts processing speed and incident cardiovascular disease beyond spindle density." An open centile calculator (age, sex, derivation -> centile) is the reusable product.
- **Venues.** *Sleep* or *SLEEP Advances* (normative + transport); *Nature Aging* / *Neurobiology of Aging* (aging biomarker framing); *Journal of the American Heart Association* or *Circulation: Cardiovascular Quality and Outcomes* (CVD arm); *Journal of Sleep Research* (methods/harmonization).
- **Follow-ups.** Longitudinal change in coupling centile (SHHS1 -> SHHS2, MrOS V1 -> V2) vs cognitive decline; wearable-EEG transfer of the centile charts; genetic correlates using NSRR family cohorts (CFS heritability of coupling).

## Risks, confounds & mitigations

- **Detector bias across amplitude/montage**: fixed-threshold detectors find fewer spindles in low-amplitude EEG (older adults, mastoid references). Mitigation: YASA's relative thresholds; report coupled fraction and MRL, which are less amplitude-dependent; per-cohort detector sensitivity analysis against manually scored subsets (DOD-H, Sleep-EDF).
- **Sampling-rate/filters**: SHHS at 125 Hz limits spindle waveform fidelity; 1990s hardware filters differ. Mitigation: harmonized 0.3-35 Hz band, resampling to 100 Hz for all cohorts, cohort random effects, and the H6 within-subject derivation test.
- **Sleep apnea and medication** confound both coupling and outcomes. Mitigation: AHI, oxygen desaturation index, benzodiazepine/antidepressant use as covariates; sensitivity analysis excluding AHI >= 30.
- **Age range imbalance** (young ages only from Sleep-EDF/CFS). Mitigation: cohort-stratified spline knots; report centile uncertainty bands; do not extrapolate beyond cohort support.
- **Survivor bias in older cohorts** (MrOS >= 65). Mitigation: inverse-probability weighting sensitivity analysis; interpret centiles as population-conditional.
- **Outcome ascertainment differs** between SHHS and MESA. Mitigation: harmonize to hard CVD (MI, stroke, CVD death); report per-cohort HRs plus meta-analytic pooling.
- **Multiple detectors/parameters = garden of forking paths**. Mitigation: parameters frozen from YASA defaults before outcome analyses; multiverse report over detector settings as supplement.

## Milestones

- [ ] Sleep-EDF + HMC downloaded (`scripts/download_data.py --dataset sleep-edf --sample`); NSRR DUA submitted for SHHS/MESA/MrOS/CFS.
- [ ] Channel maps validated per cohort (`io_edf.resolve_channel`); per-night QC table (NREM minutes, artifact %, derivation used).
- [ ] YASA spindle/SO detection over all nights; caching of event tables (`data/events/<cohort>/<night>.parquet`).
- [ ] Coupling metrics per night with surrogate normalization; detector-agreement study on DOD-H.
- [ ] Normative quantile curves (pooled and by sex); centile calculator; H1-H2.
- [ ] LOCO transportability of age predictors; harmonization ablation; H3, H6.
- [ ] Cognition models in MESA/MrOS; H4.
- [ ] SHHS1 Cox models with coupling; MESA replication; H5.
- [ ] Manuscript + open centile calculator release.

## Ethics / data-use notes

- Sleep-EDF and HMC (PhysioNet open) and DOD (Dreem) are openly licensed; cite dataset papers.
- SHHS, MESA, MrOS and CFS are distributed by NSRR under a data-use agreement: no redistribution, no re-identification attempts, cite NSRR (Zhang et al., 2018, *JAMIA*) and the parent studies, acknowledge NHLBI. MESA outcome files may require separate approval from the MESA coordinating center.
- Keep the NSRR token in the `NSRR_TOKEN` environment variable only; never commit it.
- Do not send NSRR/PhysioNet data to third-party LLM or cloud ML APIs; process locally.
- Never commit EDF/XML files; `data/` is git-ignored. Only derived, non-identifiable per-night summary tables (with NSRR ids) may be shared under the DUA terms, and only with the NSRR's permission.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py --dataset sleep-edf --sample    # two open nights + subject tables
pytest tests -q                                                 # synthetic nights, no EDF / YASA needed
```

One night end-to-end with the starter modules (YASA used automatically when installed):

```python
from spindle_age import io_edf, detect, coupling, normative, transport
night = io_edf.load_night("data/sleep-edf/sleep-cassette/SC4001E0-PSG.edf",
                          "data/sleep-edf/sleep-cassette/SC4001EC-Hypnogram.edf", cohort="sleep-edf")
sp, so = detect.detect_events(night.signal, night.fs, night.hypno_samples, backend="auto")
dens = detect.spindle_summary(sp, night.hypno, night.epoch_len)              # density-type metrics
coup = coupling.coupling_metrics(night.signal, night.fs, night.hypno_samples, sp, so)   # MRL, phase, MI, mrl_z
# after running all nights of all cohorts -> per-night metrics table `df`
model = normative.fit_normative(df.age, df.mrl, df.sex, df=5)              # sex-specific centile curves
df["mrl_centile"] = model.centile(df.age, df.mrl)
loco = transport.compare_feature_sets({"coupling": df[COUPLING_COLS].values, "density": df[DENSITY_COLS].values},
                                      df.age, df.cohort, harmonize=True, covariates=df[["sex_code"]].values,
                                      reference="density", subject=df.subject_id)
```

## Repository layout

```
README.md
requirements.txt
data/README.md             PhysioNet, NSRR (nsrr gem + token), OpenNeuro, Dreem acquisition and layout
scripts/download_data.py   open downloads, NSRR wrapper reading NSRR_TOKEN, openneuro-py wrapper
src/spindle_age/
  io_edf.py                cohort channel maps, derivation resolution/re-referencing, NSRR XML + Sleep-EDF hypnograms, preprocessing
  detect.py                YASA spindle/SO wrappers + fallback envelope/zero-crossing detectors, density summaries
  coupling.py              SO phase, coupled-spindle matching, MRL/preferred phase/Rayleigh, Tort MI, event-locked sigma, surrogate z
  normative.py             B-spline quantile regression normative model, centiles/z-scores, calibration, age-slope bootstrap
  transport.py             ComBat-style harmonizer, LOCO evaluation, paired bootstrap, feature-set comparison
tests/test_spindle_age.py  synthetic coupled vs uncoupled nights; normative recovery; LOCO with site effects
```

## Cohort variables to extract (per night)

- Identifiers: cohort, subject id, visit, night index, derivation used, original fs, device/hardware code where available.
- Demographics/covariates: age, sex, BMI, race/ethnicity (MESA), education, AHI (`nsrr_ahi_hp3u`), ODI, TST, sleep efficiency, N2/N3/REM minutes, arousal index, benzodiazepine/antidepressant use.
- Spindle metrics: density (N2, N2+N3; fast/slow), amplitude, duration, frequency, count, sigma power.
- SO metrics: rate per minute, PTP amplitude, slope, duration, SWA (0.5-4 Hz power).
- Coupling metrics: coupled fraction, preferred phase, MRL, MRL surrogate z, Rayleigh z, Tort MI, event-locked sigma up-state % increase, coupling phase SD.
- Cognition: MESA Digit Symbol Coding / CASI; MrOS 3MS / Trails B; CFS none (age norms only); Sleep-EDF none.
- Outcomes: SHHS incident CHD/CVD/stroke, CVD death, all-cause death (dates); MESA incident CVD events.

## Planned tables and figures

- Table 1: cohorts, nights, ages, sex split, derivation, fs, hardware, NREM minutes, artifact %, detector QC.
- Table 2: sex-specific normative centiles (5/25/50/75/95) of MRL and spindle density at ages 10-90 in decade steps; standardized age slopes with CIs (H1-H2).
- Table 3: LOCO transportability, rows = held-out cohort, columns = feature set x harmonization; MAE, R2, calibration slope, transport gap (H3).
- Table 4: mixed-model associations of coupling and density with processing speed in MESA and MrOS (H4).
- Table 5: Cox HRs per SD for incident CVD/CHD in SHHS1 with replication in MESA (H5).
- Figure 1: lifespan centile chart of coupling precision by sex, cohorts overlaid.
- Figure 2: within-subject derivation agreement (Fpz-Cz vs Pz-Oz; Fz-Cz vs C4-M1) for coupling vs density (H6).
- Figure 3: LOCO predicted vs chronological age per held-out cohort, coupling vs density models.
- Figure 4: Kaplan-Meier / cumulative incidence by coupling-centile tertile.

## Key references

- Purcell et al. (2017). Characterizing sleep spindles in 11,630 individuals from the National Sleep Research Resource. *Nat. Commun.* 8:15930.
- Djonlagic et al. (2021). Macro and micro sleep architecture and cognitive performance in older adults. *Nat. Hum. Behav.* 5:123-145.
- Helfrich et al. (2018). Old brains come uncoupled in sleep: slow wave-spindle synchrony, brain atrophy, and forgetting. *Neuron* 97:221-230.
- Muehlroth et al. (2019). Precise slow oscillation-spindle coupling promotes memory consolidation in younger and older adults. *Sci. Rep.* 9:1940.
- Hahn et al. (2020). Slow oscillation-spindle coupling predicts enhanced memory formation from childhood to adolescence. *eLife* 9:e53730.
- Kurz et al. (2023). The hierarchy of coupled sleep oscillations reverses with aging in humans. *J. Neurosci.*
- Chylinski et al. (2022). Timely coupling of sleep spindles and slow waves linked to early amyloid-beta burden and predicts memory decline. *eLife* 11:e78191.
- Juginovic et al. (2025). Sleep spindle density and sleep depth as predictors of cardiovascular outcomes: a prospective EEG study. *Sleep Med.*
- Adra et al. (2022). Optimal spindle detection parameters for predicting cognitive performance. *Sleep* 45:zsac001.
- Sun et al. (2019). Brain age from the electroencephalogram of sleep. *Neurobiol. Aging* 74:112-120; Paixao et al. (2020). Excess brain age in the sleep electroencephalogram predicts reduced life and healthy lifespan. *Neurobiol. Aging* 88:150-155.
- Vallat & Walker (2021). An open-source, high-performance tool for automated sleep staging. *eLife* 10:e70092 (YASA).
- Lacourse et al. (2019). A sleep spindle detection algorithm that emulates human expert spindle scoring. *J. Neurosci. Methods* 316:3-11.
- Tort et al. (2010). Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. *J. Neurophysiol.* 104:1195-1210.
- Bethlehem et al. (2022). Brain charts for the human lifespan. *Nature* 604:525-533; Rigby & Stasinopoulos (2005). Generalized additive models for location, scale and shape. *J. R. Stat. Soc. C* 54:507-554.
- Zhang et al. (2018). The National Sleep Research Resource: towards a sleep data commons. *JAMIA* 25:1351-1358.
- Kemp et al. (2000). Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE TBME* 47:1185-1194 (Sleep-EDF).
- Guillot et al. (2020). Dreem Open Datasets: multi-scored sleep datasets to compare human and automated sleep staging. *IEEE TNSRE* 28:1955-1965.
