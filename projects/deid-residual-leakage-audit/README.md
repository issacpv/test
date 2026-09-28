# deid-residual-leakage-audit

**A local-only, aggregate-count audit of residual identifier *patterns* in MIMIC-IV-Note (and MIMIC-III) with calibrated detector recall and capture-recapture estimates, run as a responsible-disclosure study for the data curators, never as an extraction of PHI.**

## Status / difficulty / timeline / compute

- Status: design + starter code (template section splitter, regex detectors that return category/section/length only, Wilson and capture-recapture estimators, planted-identifier recall calibration, small-cell suppression, offline-NER driver with CLI). No data is shipped and no text is ever written by the code.
- Difficulty: MSc-level for the regex/NER audit and statistics; the difficult parts are the governance protocol and calibrating detector recall honestly. 4-6 months.
- Compute: one credentialed, encrypted workstation. Regex over 2.6 million notes takes hours on one core; a local transformer NER over discharge summaries needs one GPU-day. No cloud, no hosted API (see Ethics).

## Background

MIMIC-IV-Note v2.2 (Johnson et al., PhysioNet 2023) contains 331,794 discharge summaries and 2,321,355 radiology reports de-identified to the HIPAA Safe Harbor standard with a transformer-based pipeline (Johnson, Bulgarelli & Pollard, ACM CHIL 2020: "Deidentification of free-text medical records using pre-trained bidirectional transformers"); protected health information (PHI) is replaced by `___` and dates are shifted consistently per patient into 2100-2200. MIMIC-III (Johnson et al., Sci Data 2016) used the rule-based `deid` pipeline (Neamatullah et al., BMC Med Inform Decis Mak 2008), which leaves typed surrogates such as `[**First Name (STitle) 123**]`. Every automated de-identifier has imperfect recall: the i2b2 2014 challenge (Stubbs & Uzuner, J Biomed Inform 2015) set the community benchmark at token-level F1 around 0.95-0.98 for the best systems, Philter (Norgeot et al., npj Digit Med 2020) reported recall above 99% on UCSF notes yet still leaves residuals at corpus scale, and Carrell et al. (JAMIA 2013) showed that realistic surrogates ("hiding in plain sight") reduce the harm of residuals but do not remove them.

At the scale of MIMIC-IV-Note, a per-token recall of 99.5% still implies thousands of residual identifier tokens. Two 2026 preprints sharpen the stakes: "Paradox of De-identification: A Critique of HIPAA Safe Harbour in the Age of LLMs" (arXiv, 2026) estimates that a 0.34% re-identification risk implies roughly 170 re-identifiable patients in a MIMIC-IV-sized corpus, and "Language Models Can Guess Your Identities from De-identified Clinical Notes" (OpenReview preprint) shows NER-style de-identification leaves inferable dependencies with protected attributes. A 2026 ACM paper on HIPAA-compliant agentic clinical AI measured policy-consistent PHI exposure and residual leakage on MIMIC-IV discharge summaries over 107,800 runs. None of these is a *curator-facing* audit: none reports which Safe Harbor classes leak, at what rate per 10,000 notes, in which template sections, with what detector recall, or how the residual profile differs between the rule-based (MIMIC-III) and transformer (MIMIC-IV) pipelines applied at the same institution.

## The research gap

What has been done (2020-2026):

- Pipeline papers report token-level recall on held-out annotated notes (Johnson et al., 2020; Norgeot et al., 2020) but not the corpus-level residual rate of the released data.
- Re-identification-risk papers (arXiv 2026; OpenReview preprint) estimate *inference* risk from context using LLMs; they do not count mechanical residuals and, by construction, cannot be run without sending notes to models.
- The agentic-AI HIPAA architecture paper (ACM 2026) measures leakage of a downstream system, not of the corpus.
- Community folklore (GitHub issues, mailing lists) reports isolated residuals in MIMIC notes, but there is no systematic, calibrated count.

What is specifically missing (our angle):

1. **Calibrated rates per Safe Harbor class** (phone, email, URL, SSN-like, MRN-like, ZIP/street, unshifted dates, ages > 89, names in title/signature contexts, institution names, provider/device identifiers) per 10,000 notes, by note type, template section, era (`anchor_year_group` via `hadm_id`), service and note length, each with Wilson CIs and detector recall measured on planted synthetic identifiers and on open gold corpora.
2. **Capture-recapture between independent detectors** (regex vs local NER vs a locally run i2b2-trained model) to estimate how many residual-bearing notes *no* detector finds (Chapman estimator), a statistic the de-identification literature has not reported.
3. **Where residuals live**: the placeholder-density-normalised residual rate per section (signature blocks, `Facility:` lines, follow-up instructions, addenda, radiology `NOTIFICATION:` lines), which tells curators exactly which templates to target.
4. **Method comparison at one institution**: MIMIC-III (rule-based `deid`, 2001-2012) vs MIMIC-IV-Note (transformer, 2008-2019) residual profiles, unpaired, by class. This is the only public setting where two de-identification generations were applied to notes from the same hospital.
5. **A reusable, reviewable audit protocol** (counts only, cells >= 10, disclosure to the curators before publication) that other credentialed corpora (eICU notes, n2c2 releases, national EHR extracts) can adopt.

## Research questions / hypotheses

1. **RQ1 (rates).** What is the calibrated residual-candidate rate per class? H1: unshifted dates and title/signature-context names are the most frequent classes (> 5 per 10,000 discharge summaries after recall correction); contact identifiers (phone, email, URL) are rarer (< 1 per 10,000) but non-zero.
2. **RQ2 (localisation).** H2: > 60% of residual candidates in discharge summaries fall in three sections (`Followup Instructions`, `Discharge Instructions`, `Facility`/signature blocks), and the residual rate per placeholder (`___`) is highest in sections with the fewest placeholders (the model under-fires where PHI is rare).
3. **RQ3 (note type and era).** H3: radiology reports have a lower per-note but higher per-token residual rate than discharge summaries; rates decline over eras (template standardisation), with a discontinuity at the 2008-2010 to 2011-2013 boundary.
4. **RQ4 (undetected residuals).** H4: Chapman estimates from regex-vs-NER overlap imply that single-detector counts miss 30-50% of residual-bearing notes for name-like classes and < 10% for numeric classes.
5. **RQ5 (pipeline generation).** H5: MIMIC-IV-Note has a lower residual rate than MIMIC-III for names and dates, but a *higher* relative share of institution and provider identifiers (classes the transformer saw rarely in training).

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV-Note v2.2 | Primary audit corpus: `discharge.csv.gz`, `radiology.csv.gz` (+ `_detail`) | 331,794 discharge summaries; 2,321,355 radiology reports | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimic-iv-note/2.2/ |
| MIMIC-IV v3.1 `hosp` | Era (`anchor_year_group`), service, demographics for stratified rates via `hadm_id` | small tables | PhysioNet credentialed | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-III v1.4 `NOTEEVENTS` | Rule-based-pipeline comparison corpus (typed surrogates) | 2,083,180 notes | PhysioNet credentialed | https://physionet.org/content/mimiciii/1.4/ |
| PhysioNet `deid` gold-standard corpus | Detector recall on realistic surrogate PHI in nursing notes | ~2.4k notes | Open | https://physionet.org/content/deid/1.1/ |
| i2b2/n2c2 2014 de-identification corpus | Second recall gold standard (surrogate PHI, 7 classes) | 1,304 records | DUA, free registration (n2c2 portal) | https://portal.dbmi.hms.harvard.edu/ |
| Planted synthetic identifiers (this repo) | Recall calibration with obviously fictitious values (555-01xx phones, example.com) | any | Generated locally | - |

## Methods

Pipeline (each step maps to a module in `src/deid_audit/`):

1. **Sectioning** (`sections.py`). Discharge summaries are split at the fixed MIMIC-IV template headers (`Name:` ... `Followup Instructions:`), radiology reports at `EXAMINATION:` ... `NOTIFICATION:`. Placeholder statistics (`___` per section; MIMIC-III surrogate type histogram) are computed per note.
2. **Detector arm A: regex** (`patterns.py`). One pattern per Safe Harbor class; findings carry category, section and match length only. Template headers are excluded by construction. `date_shifted` (years 2100-2200) is counted separately as a positive control of the shift.
3. **Detector arm B: local NER** (`audit.py`). spaCy `en_core_web_trf`/`_sm` label histograms (PERSON, GPE, LOC, FAC, ORG, DATE), and optionally a locally cached i2b2-trained de-identification model or Philter, run offline. Entity strings are discarded at the tokenizer boundary.
4. **Per-note count rows** (`audit.aggregate_notes`): category counts, per-section counts, presence flags, placeholder stats, NER histograms, note length. This table contains no text and is the only artefact stored.
5. **Calibration** (`estimate.py`). Recall per class from (a) planted fictitious identifiers inserted into clean notes (`plant_synthetic_phi`), (b) the `deid` gold corpus and (c) n2c2 2014; precision from a fixed-size, IRB-approved manual review of *categories* on the credentialed machine by two credentialed reviewers (kappa reported; no strings copied out).
6. **Estimation.** Note-level rates per 10,000 with Wilson CIs (`rate_per_10k`), recall-corrected rates (`calibrated_rate`), Chapman capture-recapture between arms A and B per class (`capture_recapture_from_flags`), section-level rates normalised by placeholder density, era/service/note-length regressions (Poisson GLM with note length offset, statsmodels).
7. **Method comparison.** Same detectors on MIMIC-III `NOTEEVENTS` discharge summaries and radiology reports; residual-per-surrogate ratios by class.
8. **Disclosure.** A curator report (class x section x era counts, detector recipes, recommended template fixes) goes to the PhysioNet/MIT-LCP team before any submission; only cells >= 10 appear in the paper.

Tools: Python `re`, pandas, statsmodels; spaCy (local); optional Philter / Presidio / a cached transformer de-identifier; no network access during the audit.

### Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests                                    # synthetic notes only
python scripts/download_data.py --dataset deid-gold --out data/deid-gold      # open calibration corpus
PYTHONPATH=src python -m deid_audit.audit --input data/mimic-iv-note/2.2/discharge.csv.gz \
    --note-type discharge --ner-model en_core_web_sm --out outputs/discharge_counts.csv
```

## Evaluation & statistics

- Unit of analysis: the note (presence flags) for rates and capture-recapture; the token for per-token rates.
- Recall calibration: planted identifiers (10 per class per 1,000 clean notes, 5 seeds), `deid` gold corpus, n2c2 2014; report recall per class with Wilson CIs and propagate it into rate CIs by parametric bootstrap.
- Precision: two credentialed reviewers classify a stratified random sample of 100 candidates per class *on the approved machine*, recording only true/false per category; Cohen's kappa; strings are never copied.
- Capture-recapture assumptions (independence, closed population, equal catchability) are examined by class; where arms are correlated (both keyed on capitalisation), report Chapman as a lower bound and use a third arm (transformer model) for a log-linear three-source model.
- Multiple comparisons: classes x note types x eras, Benjamini-Hochberg; H1-H5 pre-registered.
- Nulls and controls: (i) shifted-date detector must fire in ~100% of discharge summaries (positive control of the pipeline); (ii) synthetic clean notes must yield zero candidates (tests enforce this); (iii) shuffled-token notes give the false-positive floor of each regex.
- Reporting: only aggregate counts, cells < 10 suppressed (`suppress_small_cells`), no examples, no offsets.

## Publishable angle

Headline result: "Across 2.65 million de-identified MIMIC-IV notes, residual identifier candidates occur at X per 10,000 notes, dominated by unshifted dates and title-context names, concentrated in follow-up and signature sections; a two-detector capture-recapture estimate implies Y% of residual-bearing notes escape any single detector, and the transformer pipeline reduced name/date residuals but not institution residuals relative to the rule-based pipeline." The deliverable that makes the paper is the calibrated audit protocol and the curator report, not the numbers themselves.

Target venues: Journal of the American Medical Informatics Association; Journal of Biomedical Informatics; npj Digital Medicine; ACM CHIL (proceedings) for the methodology; a short report to PhysioNet as the disclosure artefact.

Follow-ups: apply the protocol to eICU and n2c2 releases; use section-level findings to fine-tune a local de-identifier and measure the residual drop; study quasi-identifier density (occupation, rare-event mentions) as a distinct, aggregate-only risk metric.

## Risks, confounds & mitigations

- Regex false positives (drug names after "Dr.", "St." for saint, 20xx in device model numbers): measured precision per class from the on-machine review; report calibrated rates, not raw counts.
- Detector correlation violates capture-recapture independence: three-source log-linear models; report bounds.
- Reviewer exposure to residual PHI: the review protocol records categories only, on the approved machine, under the DUA's research purpose; reviewers are credentialed and the protocol is IRB-reviewed.
- Publication could itself help re-identification: no strings, no offsets, no note ids; suppressed cells; disclosure to curators first with an agreed embargo.
- MIMIC-III and MIMIC-IV notes overlap in era but cannot be paired without linking identifiers; the comparison is unpaired, and no attempt is made to link records across datasets.
- Template drift across eras confounds section rates: stratify by era and by detected template version (header set).

## Milestones

- [ ] IRB/ethics consultation on the audit protocol; PhysioNet credentialing (MIMIC-IV-Note, MIMIC-IV, MIMIC-III)
- [ ] Open `deid` gold corpus and n2c2 2014 obtained; planted-identifier recall calibration run for all classes
- [ ] Regex arm over all discharge and radiology notes; per-note count table
- [ ] Local NER arm (spaCy, optional transformer de-identifier) over discharge summaries
- [ ] Precision review (two reviewers, categories only, kappa)
- [ ] Rate tables per class x section x era x note type; Poisson GLMs; capture-recapture estimates
- [ ] MIMIC-III comparison run
- [ ] Curator report delivered to PhysioNet; embargo agreed
- [ ] Pre-registration (OSF); manuscript with aggregate tables only

## Ethics / data-use notes

- MIMIC-IV-Note, MIMIC-IV and MIMIC-III are PhysioNet credentialed resources: CITI training, signed DUAs, approved encrypted storage, no redistribution.
- PhysioNet's responsible-use policy prohibits sharing credentialed data with third parties and restricts the use of online services; this project sends **no data to any hosted API or LLM** and uses only local tooling (`re`, spaCy models installed on the machine, optional locally cached de-identification models with network access disabled).
- The DUA prohibits attempts to identify individuals. This audit never attempts identification: it counts pattern *classes*, stores no matched text, no offsets and no note identifiers with findings, suppresses cells below 10, and does not link records across datasets.
- Responsible disclosure: findings are reported to the PhysioNet/MIT-LCP curators before publication, with an agreed embargo, so that the corpus can be corrected.
- Never commit data or count tables derived from it; `.gitignore` excludes `data/` and `outputs/`. Credentials come from `PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD`.
