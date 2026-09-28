# neurofeedback-multiverse: would you have been rewarded? A specification-curve re-analysis of the feedback signal in open EEG neurofeedback datasets

**One-sentence pitch.** Recompute the *feedback signal itself* - not just the offline outcome - of open EEG neurofeedback (NF) datasets on OpenNeuro under a full specification space of the online pipeline (reference, band definition, power estimator, window, baseline normalisation, artifact handling, smoothing), and quantify how many reward decisions would have flipped, how unstable the "learner / non-learner" label is, how much of the delivered feedback was ocular or muscular artifact, and whether the contingency between feedback and the intended neural target predicts learning.

## Status / difficulty / timeline / compute

- Status: design + working starter code (this repo): specification space (2,160 pipelines), feedback recomputation, reward-decision agreement, learning indices, learner-label stability, feedback-specificity and contingency metrics, and the joint specification-curve permutation test, all tested on synthetic EEG with a learning alpha oscillator, blinks, saccades and EMG bursts.
- Difficulty: MSc thesis (signal processing + meta-science statistics). No model training; the challenge is careful handling of each dataset's timing/marker conventions and the inferential framework for the multiverse.
- Timeline: 4-7 months (1 month data screening + manifests, 1 month MR-artifact / preprocessing fixed steps for the EEG-fMRI sets, 2 months multiverse runs + statistics, 1-2 months writing).
- Compute: CPU only. 2,160 specifications x ~100 sessions x a few minutes of EEG each is hours on a workstation; the artifact-regressor and Hilbert estimators dominate.

## Background

EEG neurofeedback trains participants to change a band-power (alpha, SMR, frontal-midline theta, mu ERD) using a signal computed *online* from a few electrodes. Roughly a third of participants do not learn (Alkoby et al., 2018, *Neuroscience*; Weber, Ethofer & Ehlis, 2020, *NeuroImage: Clinical*), sham-controlled trials often find no specific effect (Schabus et al., 2017, *Brain*; Thibault, Lifshitz & Raz, 2016, *Cortex*), and learning is a skill that need not produce clinical benefit (Micoulaud-Franchi et al., 2021, *Front. Hum. Neurosci.*). The CRED-nf checklist (Ros et al., 2020, *Brain*) therefore asks authors to report how the feedback signal was computed and to demonstrate that the target was actually regulated, because the number the participant sees is the product of at least seven engineering choices that papers rarely report in full (Enriquez-Geppert, Huster & Herrmann, 2017, *Front. Hum. Neurosci.*; Sitaram et al., 2017, *Nat. Rev. Neurosci.*).

Two developments make a re-analysis possible now: open NF datasets on OpenNeuro in BIDS-EEG with training blocks marked (the simultaneous EEG-fMRI motor-imagery NF sets XP1/XP2, Lioi et al., 2020, *Sci Data*; two 2025 datasets of parietal-alpha down-regulation NF in immersive VR, *Front. Neurosci.* 2025), and the multiverse / specification-curve framework (Steegen et al., 2016, *Perspect. Psychol. Sci.*; Simonsohn, Simmons & Nelson, 2020, *Nat. Hum. Behav.*), which has been applied to EEG preprocessing for ERPs and decoding but not to the closed-loop signal.

## The research gap

**What has been done (2023-2026):**

- A mega-analysis of frontal-midline-theta NF (*NeuroImage*, 2026; five studies, N = 168, raw participant-level data) characterised learning trajectories (session-to-session and within-session indices) and compared a standard 4-8 Hz band with an individualised theta band - a two-level comparison of one factor, for one protocol.
- Multiverse analyses of EEG *preprocessing* exist for cognitive tasks ("No single best pipeline", *Psychophysiology*, 2025; "How EEG preprocessing shapes decoding performance", arXiv:2410.14453, 2024; "Same brain, different prediction", arXiv:2605.07212, 2026), showing that pipeline choices change conclusions. None concerns a feedback signal, reward decisions or learner classification.
- Predictors of NF learning (resting alpha amplitude, traits, sex) are studied with a *fixed* learning index per paper; whether the index itself is robust is not asked.
- Feedback-signal contamination by EOG/EMG is acknowledged in reviews and in the CRED-nf checklist, but no dataset-level quantification of how much delivered feedback was artifact exists.

**What is missing (this project):**

1. **Reward-decision agreement.** The fraction of online reward decisions that would differ under an alternative, equally defensible pipeline has never been measured. It bounds the "dose" of contingent reinforcement a protocol actually delivered.
2. **Learner-label stability across the multiverse.** The ~30% non-learner rate is treated as a participant property; if it depends on the analysis pipeline, predictor studies are studying pipeline x participant interactions.
3. **Feedback specificity.** Variance of the online signal explained by ocular/muscular regressors per dataset, per specification, and its relation to learning.
4. **Contingency as a mechanism.** Whether the dependence between delivered feedback and a cleaner offline estimate of the target (or the ground truth in simulation) predicts who learns, across datasets.
5. A **joint inferential statement**: is "participants learned to regulate the target" robust over the whole specification space (specification-curve permutation test), or does it hold only in some corners?

## Research questions / hypotheses

1. **H1 (decision agreement).** Median pairwise agreement of binary reward decisions across specifications is < 0.80 in every dataset; agreement is lowest when the band definition (fixed vs individual alpha) or the artifact handling changes, and highest across estimators (Welch vs Hilbert vs FFT, > 0.85).
2. **H2 (learner instability).** >= 25% of participants change learner / non-learner status across specifications (label unstable in 25-75% of specs); mean pairwise Cohen's kappa of learner labels across specifications is < 0.6.
3. **H3 (specificity).** Ocular + muscular regressors explain >= 20% of feedback variance in at least one dataset under the recording-reference / no-artifact-handling specification, and < 10% under Laplacian + threshold specifications; the specificity of a participant's online signal correlates negatively with their learning index (Spearman rho < -0.3).
4. **H4 (robust direction, fragile size).** The sign of the mean learning slope is stable (>= 90% of specifications) for the trained band in the alpha datasets, but its magnitude varies by more than a factor of two across the curve; the joint permutation test is significant for the target band and not for a control band (beta 15-25 Hz).
5. **H5 (contingency predicts learning).** Feedback-target contingency (Spearman rho or mutual information between delivered feedback and the Laplacian + EOG-regressed offline estimate of the same band) predicts the participant's learning slope across datasets (rho > 0.3), after adjusting for baseline band power.
6. **H6 (direction asymmetry).** Down-regulation protocols (VR alpha datasets) show lower reward-decision agreement than up-regulation (motor-imagery ERD) because artifact-driven power increases are rewarded in one direction and punished in the other.

## Datasets

| Dataset | What is used | Size | Access | URL |
|---|---|---|---|---|
| OpenNeuro ds002336 (XP1; Lioi et al., 2020, Sci Data) | 64-ch EEG during unimodal EEG-NF, fMRI-NF and bimodal motor-imagery NF; online NF scores per data descriptor | 10 subjects, 1 session, several NF runs; EEG + fMRI + T1 (GBs) | Open (OpenNeuro, no login) | https://openneuro.org/datasets/ds002336 |
| OpenNeuro ds002338 (XP2; Lioi et al., 2020, Sci Data) | Bimodal EEG-fMRI motor-imagery NF, 1-D vs 2-D visual feedback | 20 subjects, 3 NF runs | Open | https://openneuro.org/datasets/ds002338 |
| OpenNeuro ds005846 and ds005878 (Front. Neurosci., 2025) | EEG near-real-time parietal-alpha *down*-regulation NF in immersive VR; EEG-only | see dataset pages (2025 releases) | Open | https://openneuro.org/datasets/ds005846 , https://openneuro.org/datasets/ds005878 |
| Further OpenNeuro EEG-NF datasets found by `scripts/download_data.py --scan` | any EEG-based NF with marked blocks | varies | Open | https://openneuro.org/search/modality/eeg |
| Frontal-midline-theta mega-analysis data (NeuroImage, 2026) | optional replication target (5 studies, N = 168) | 168 participants | On request from the authors (verify availability) | see paper |

Selection rule: EEG-based feedback, continuous raw EEG, training blocks recoverable from `events.tsv` or the paper, feedback direction and electrode(s) documented. Datasets that ship the online feedback values are the primary set (they allow the *online vs recomputed* comparison); others enter the multiverse-only analyses.

## Methods

1. **Fixed pre-steps (outside the multiverse).** MR gradient/BCG artifact correction for XP1/XP2 (as in the descriptor), resampling to 250 Hz, channel-name harmonisation, block/baseline timing from `events.tsv` into `derived/manifests/sessions.csv`. These are applied identically to every specification and documented once.
2. **Specification space** (`nf_multiverse.specs`): reference {none, CAR, Laplacian, linked mastoids} x band {fixed, individual alpha frequency from the baseline} x estimator {Welch, periodogram, Hilbert envelope, band-pass RMS} x window/step {0.5/0.125, 1/0.25, 2/0.5 s} x normalisation {baseline z, ratio, log-ratio} x artifact handling {none, amplitude-threshold freeze with hold-last, EOG regression} x smoothing {1, 4 windows} = 2,160 specifications; per dataset the *documented online pipeline* is one point in this space and is flagged.
3. **Feedback recomputation** (`nf_multiverse.feedback`): `compute_feedback()` returns the feedback time series, raw band power, the band actually used and the frozen fraction; `reward_decisions()` implements baseline-threshold, percentile and adaptive-median reward rules (direction up/down); `agreement_matrix()` resamples different window grids onto a common one.
4. **Learning indices** (`nf_multiverse.learning`): within-session slope of block means, session-to-session slope, learner classification (significant slope in the trained direction), `learner_agreement()` (pairwise kappa, unanimous / unstable fractions, learner-rate range), `specification_curve()` with participant bootstrap CIs, and `specification_curve_test()` (joint sign-flip permutation on median effect, share positive, share significant).
5. **Specificity and contingency** (`nf_multiverse.artifacts`): per-window log power of frontal-polar 0.5-4 Hz (EOG), lateral 15-100 Hz (EMG) and broadband RMS as nuisance regressors on the feedback grid; `feedback_specificity()` gives R^2 and standardised betas; `contingency()` gives Spearman rho, plug-in mutual information and the reward-target point-biserial correlation.
6. **Synthetic validation** (`nf_multiverse.synthetic`): 1/f background, alpha oscillator with block-wise learning gain and slow drift, Gaussian blinks, step-like saccades (1/f^2 leakage into alpha), broadband EMG bursts; used for unit tests and for a power analysis of H2/H5 at the real datasets' n.
7. **Statistics**: mixed models (participant random intercept, dataset fixed effect) for H3/H5; specification-curve tests per dataset and per band (target vs control band); reward-decision agreement summarised as median and IQR over specification pairs, and decomposed by factor with a fractional-factorial ANOVA on the agreement matrix.

Tools: `numpy`, `scipy`, `pandas`; `mne` + `mne-bids` for reading; `openneuro-py` or the public S3 bucket for download; `statsmodels` (optional) for mixed models.

## Evaluation & statistics

- Unit of analysis: participant x session x specification. Learning effects are participant-level slopes; inference is over participants (bootstrap and sign-flip permutation), never over windows.
- Pre-registration: the specification space, the learner definition (significant positive slope, alpha = 0.05) and H1-H6 are frozen before the real data are processed; exploratory factors added later are labelled as such.
- Multiplicity: the specification-curve test is a single joint test per dataset x band; H3/H5 correlations are corrected with Holm within their family; agreement and kappa summaries are descriptive with bootstrap CIs.
- Nulls: (i) control band (beta) and control electrodes (Fz for a Pz protocol) must show no learning; (ii) block-order shuffling within participant for the slope null; (iii) sign-flip null for the specification curve; (iv) simulated sessions with `learning_gain = 0` as the false-positive check for the whole pipeline.
- Leakage-type pitfalls specific to this design: individual alpha frequency is estimated from the *baseline only*; artifact thresholds are fixed a priori (100 microvolts) and not tuned on outcomes; the online pipeline's own specification is analysed without knowledge of which participants were reported as learners.
- Robustness: repeat with the majority-vote learner definition and with slope > 0 regardless of significance; report results with and without the frozen windows.

## Publishable angle

- **Headline.** "Across N open neurofeedback sessions, one in four reward decisions depends on undocumented pipeline choices, one in three participants changes learner status across defensible pipelines, up to X% of the delivered feedback variance was ocular/muscular, and feedback-target contingency - not the reported learning index - predicts who learns." Delivered with a tool that any lab can run on its own raw data before publishing a learner rate.
- **Venues.** *NeuroImage* or *Human Brain Mapping* (methods + re-analysis); *Journal of Neural Engineering* (closed-loop engineering); *Psychophysiology* (multiverse tradition); *Imaging Neuroscience*; a short report in *Brain* / *Clinical Neurophysiology* if the specificity results are strong.
- **Follow-ups.** Add fMRI-NF datasets (ROI/GLM multiverse); simulate the *behavioural* consequence of decision flips with a reinforcement-learning model of the participant; real-time implementation of the most specific pipeline (Laplacian + EOG regression + IAF) and a sham-controlled test; extension to the FM-theta mega-analysis data.

## Risks, confounds & mitigations

- **Few datasets ship the online feedback values.** Mitigation: the multiverse and learner-stability analyses do not need them; the online-vs-recomputed comparison is reported for the subset that does; the documented online pipeline is reproduced as a point in the space and its agreement with the shipped values (when present) validates the reproduction.
- **EEG-fMRI residual artifacts (XP1/XP2)** could dominate specificity. Mitigation: fixed, documented MR-artifact correction; results reported separately for the scanner and VR datasets; H3 pre-registered on the EEG-only datasets.
- **Block timing errors** inflate or destroy slopes. Mitigation: manifests built from `events.tsv` and checked against the descriptors; sensitivity to +-2 s shifts.
- **Motor-imagery ERD is a *decrease*; alpha-VR is a decrease; FM-theta is an increase.** Mitigation: `direction` is explicit everywhere; H6 tests the asymmetry.
- **Small n per dataset (10-20).** Mitigation: participant-level bootstrap; power analysis on synthetic sessions; pooled mixed models with dataset effects; conclusions framed per dataset and pooled.
- **Garden of forking paths in the multiverse itself.** Mitigation: the space is pre-registered and *complete* (all combinations run), not curated after seeing results.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests                       # 8 tests on synthetic sessions (learning, IAF, agreement, artifacts, spec-curve test)
python scripts/download_data.py --scan                         # OpenNeuro GraphQL: candidate EEG-NF datasets -> data/candidates.csv
python scripts/download_data.py --dataset ds002338 --sample    # public S3 bucket over HTTPS
```

Minimal multiverse on one recording (numpy arrays from `mne`; see data/README.md):

```python
from nf_multiverse import build_specification_space, compute_feedback, reward_decisions, within_session_slope
from nf_multiverse.learning import block_means
specs = build_specification_space(max_specs=200)
rows = []
for s in specs:
    fb = compute_feedback(eeg, ch_names, fs, s, target_channels=("Pz",), baseline=(0, 60))
    res = within_session_slope(block_means(fb.times, fb.values, block_edges))
    rows.append({"spec": s.name(), "slope": res["slope"], "p": res["p"], "reward_rate": reward_decisions(fb.values).mean()})
```

## Repository layout

```
src/nf_multiverse/
  specs.py       FeedbackSpec (7 factors) and build_specification_space(); spec_table()
  feedback.py    extract_target (none/CAR/Laplacian/mastoid), IAF, band_power_windows (welch/fft/hilbert/rms), normalisation, artifact handling, compute_feedback, reward_decisions, agreement_matrix
  learning.py    block_means, within/session slopes, classify_learner, cohen_kappa, learner_agreement, specification_curve, specification_curve_test (joint sign-flip)
  artifacts.py   artifact_regressors (EOG/EMG/RMS on the feedback grid), feedback_specificity (R^2), contingency (Spearman, MI, reward-target correlation)
  synthetic.py   simulate_session: 1/f background, learning alpha oscillator, blinks, saccade steps, broadband EMG bursts
scripts/download_data.py   OpenNeuro discovery (GraphQL) and download (S3 ListObjectsV2 over HTTPS)
tests/test_nf_multiverse.py
```

## Analysis plan

| Hypothesis | Unit | Statistic | Inference | Control |
|---|---|---|---|---|
| H1 decision agreement | spec pair x session | median pairwise agreement; factor decomposition | bootstrap over sessions; fractional-factorial ANOVA on agreement | agreement between identical specs = 1 |
| H2 learner instability | participant x spec | pairwise kappa, unstable fraction, learner-rate range | bootstrap over participants | simulated sessions with known learners |
| H3 specificity | participant x spec | R^2 of feedback on EOG/EMG/RMS regressors; rho(R^2, slope) | mixed model (participant, dataset) | Laplacian + threshold spec as low-artifact reference |
| H4 direction robust, size fragile | dataset x band | specification curve; share positive; median | joint sign-flip permutation | beta control band; block-order shuffle |
| H5 contingency predicts learning | participant | rho(contingency, slope) adjusted for baseline power | mixed model; Holm within family | learning_gain = 0 simulations |
| H6 direction asymmetry | dataset | agreement in down- vs up-regulation protocols | bootstrap difference | simulated up/down sessions |

## Milestones

- [ ] `--scan` OpenNeuro; screen candidates; freeze the dataset list and the pre-registration (OSF).
- [ ] Download XP1/XP2 and the VR datasets; build `sessions.csv` manifests; fixed pre-steps.
- [ ] Reproduce each dataset's documented online pipeline as a specification; compare with shipped online values where available.
- [ ] Run the 2,160-specification multiverse on all sessions; store feedback per spec.
- [ ] Reward-decision agreement matrices; factor decomposition; H1/H6.
- [ ] Learning indices, learner labels, stability; specification curves and joint tests; H2/H4.
- [ ] Specificity and contingency analyses; mixed models; H3/H5.
- [ ] Synthetic power analysis; robustness checks; manuscript; release the tool with a CLI.

## Ethics / data-use notes

- All datasets are de-identified, openly licensed OpenNeuro resources; cite the data descriptors and OpenNeuro. No credentialed data are involved.
- Do not commit raw or derived per-participant EEG; `data/` and `derived/` (except manifests) are git-ignored.
- Learner / non-learner labels are analysis constructs, not participant traits; results are reported at the aggregate level and never as re-identifying per-participant claims about the original studies' conclusions.
- Re-analyses that contradict a dataset's original report are communicated to the original authors before publication, in line with the CRED-nf spirit of constructive reporting.
