# unlabeled-adverse-event-mining — a systematic unlabeled-signal map and label-lag analysis for US drug labels

**Pitch.** Extract the adverse reactions that every US prescription label already lists (SPL Boxed Warning, Warnings & Precautions, Adverse Reactions incl. postmarketing), compute FAERS disproportionality for the events that are *not* on the label, date the first emergence of each signal with a quarterly time scan, measure the lag between signal emergence and the label update that eventually named it, and evaluate the whole pipeline against FDA's own quarterly "potential signals" list.

**Status:** starter code + protocol. **Difficulty:** MSc–PhD; **timeline:** 6–9 months (12 with full label-version history). **Compute:** laptop for the count-based scan; full FAERS (bulk JSON, ~60 GB) and all SPL versions need a workstation with ~200 GB disk; no GPU unless scispaCy/transformer NER is used.

## Background

Post-marketing signal detection in FAERS is meant to find risks that are not yet on the label, yet the standard workflow (ROR/PRR/IC on the whole database) does not know what the label says. Labelled adverse reactions therefore dominate published "signals", the truly new ones are buried, and the time between when a signal became detectable and when the label changed is rarely measured. The regulatory record shows this lag is long: Dhodapkar et al. (2022, *BMJ*) found FDA posted 603 FAERS-derived potential signals between 2008 and 2019, 68.5% resolved, three quarters of resolutions being labelling changes — but the study starts the clock at FDA's listing, not at signal emergence. Automated extraction of labelled reactions from SPL is now feasible: ETHER-based extraction of Adverse Reactions sections (2018, *Health Informatics J*), the 200-label ADR annotation corpus (Demner-Fushman et al., 2018, *Sci Data*), and PVLens (2025, arXiv:2503.20639), which maps SPL terms to MedDRA for 5 358 substances with recall 0.98 / precision 0.80 against expert review.

## The research gap

**Done.**
- Label extraction resources: ETHER pipeline; the TAC 2017 ADR-from-label task and Demner-Fushman et al. (2018) corpus; PVLens (2025) as a validated, current labelled-ADR database.
- FAERS disproportionality at scale, including time-scan / "signal emergence" methods in VigiBase and OMOP-style evaluations, and Bayesian borrowing across semantically similar events (2025, arXiv:2504.12052).
- Regulatory audits: Dhodapkar et al. (2022, *BMJ*) on FDA's potential-signals list; historical estimates of median time to boxed-warning additions of several years; a 2025 estimate of a large "labelling gap" for off-patent drugs.
- Hundreds of single-drug FAERS papers that compare a few signals with the label by hand.

**Missing — and what this project delivers.**
1. A **database-wide unlabeled-signal map**: for every US prescription ingredient, every FAERS PT with a sustained disproportionality signal, annotated on/off label using an automated SPL extractor (PVLens-style, reproduced here with an open dictionary + optional scispaCy) — no prior work joins a *validated labelled-ADR set* with a *dated* FAERS scan across all drugs.
2. **Signal-emergence dating and label lag**: first sustained-signal quarter from cumulative quarterly 2x2 tables, joined to the first label version naming the term (DailyMed version history / FDA SrLC) → Kaplan–Meier time-to-labelling and its predictors (seriousness, drug age, generic status, reporter mix). Prior lag estimates start at FDA's listing or approval, not at emergence.
3. **Evaluation against FDA's own list** as ground truth: sensitivity and lead time of the scan for the quarterly potential-signals entries 2008–2025, plus a precision proxy — a reproducible benchmark that later methods (Bayesian borrowing, LLM triage) can be scored on.
4. **Bias-aware ranking**: signals ranked by strength, recency and seriousness with reporter-type and stimulated-reporting sensitivity analyses (shared conventions with the sibling `faers-reporting-bias` project).

## Research questions / hypotheses

1. **H1 (unlabeled share).** Among sustained FAERS signals (IC025 > 0, n ≥ 3 for ≥ 2 consecutive quarters) for ingredients with ≥ 1 000 reports, ≥ 25% concern PTs absent from all sections of the current label after synonym/hierarchy mapping; the unlabeled share is higher for drugs approved > 15 years ago and for generics-only ingredients.
2. **H2 (lag).** For signal–label pairs where the label was updated after the signal, the median lag is > 24 months; lags are shorter for serious/designated medical events (DME list) and for drugs still under NDA exclusivity.
3. **H3 (FDA benchmark).** The scan detects ≥ 60% of FDA potential-signal entries (2008–2025) before or in the quarter FDA posted them, with a median lead of ≥ 2 quarters; sensitivity is lower for rare, serious events flagged from case-series review than for high-volume events.
4. **H4 (label-first vs signal-first).** A substantial fraction (> 30%) of FAERS signals for labelled events emerges *after* labelling (label-driven / stimulated reporting), which must be excluded when estimating lag — quantifying this artefact is itself a result.
5. **H5 (extractor validity).** The open extractor reaches ≥ 0.85 F1 against 100 manually annotated labels (and against PVLens where overlapping), and residual errors do not change H1–H3 conclusions (sensitivity analysis with PVLens output where available).

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| openFDA `drug/label` (current SPLs) | labelled ADR extraction | ~45 k Rx set ids; ~1 GB zipped | open | https://open.fda.gov/apis/drug/label/ |
| DailyMed web services + SPL archive | label version history and archived versions | all versions since 2005 | open | https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm |
| FDA Safety-related Labeling Changes (SrLC) database | dated safety label changes with sections & summaries (2016→) | ~5 k entries | open | https://www.accessdata.fda.gov/scripts/cder/safetylabelingchanges/ |
| FAERS via openFDA `drug/event` (API counts + bulk JSON) | quarterly drug×PT counts; time scan | ~20 M reports; ~60 GB | open | https://open.fda.gov/apis/drug/event/ |
| FDA quarterly "Potential Signals of Serious Risks" pages | ground truth for evaluation | ~50 entries/yr since 2008 | open (HTML tables) | https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers/potential-signals-serious-risksnew-safety-information-identified-fda-adverse-event-reporting-system |
| MedDRA (PT/LLT/HLT ASCII) | term normalisation and hierarchy | ~80 k LLTs | licensed (free for academia via MSSO) | https://www.meddra.org/ |
| PVLens labelled-ADR database (if released) / SIDER 4.1 | external validation of extraction | 5 358 substances / 1 430 drugs | open (check licences) | https://arxiv.org/abs/2503.20639 ; http://sideeffects.embl.de/ |
| Demner-Fushman et al. 2018 annotated labels (200 SPLs) | extractor training/validation | 200 labels | open | https://www.nature.com/articles/sdata20181 |

Acquisition: `data/README.md`; sample: `python scripts/download_data.py --sample`.

## Methods

1. **Labels.** Pull current Rx SPLs (`label_client.LabelClient.labels_for_generic` / bulk), parse safety sections and the postmarketing subsection (`parse_label`, `split_subsections`); pool set ids per ingredient and keep the RLD as primary.
2. **Term extraction.** Dictionary matcher with negation and plural/spelling handling (`term_extractor.extract_terms`), MedDRA LLT→PT via `MedDRADictionary.from_meddra_ascii`; optional scispaCy `en_ner_bc5cdr_md` candidates mapped to PTs (`scispacy_candidates`, `map_candidates`). Output: per-ingredient labelled PT set with section and `postmarketing_only` flag (`extract_from_label`, `labeled_terms`). Validate on 100 annotated labels + Demner-Fushman corpus.
3. **FAERS scan.** Build (report, quarter, suspect drug, PT) from bulk JSON, de-duplicated by `safetyreportid`; cumulative quarterly 2x2 tables and ROR/PRR/IC (`signal_scan.cumulative_pair_tables`, `disproportionality`); first sustained signal per pair (`time_scan`, k_sustain = 2, min n = 3); recency = reports in the last 4 quarters. Count-only fallback from openFDA `count` queries (`cumulative_from_counts`).
4. **Unlabeled flagging & ranking.** `flag_unlabeled` (exact / synonym / HLT-sibling match through MedDRA hierarchy) and `rank_unlabeled` (IC025 + recency + seriousness). Sensitivity: HCP-only reports; exclude 12 months after FDA communications.
5. **Label lag.** For labelled pairs, first version containing the PT from DailyMed history + SPL XML diffs (`parse_spl_xml_sections`) or SrLC summaries (`label_lag.srlc_terms`); `compute_label_lag` → categories signal-first / label-first / censored; `kaplan_meier` and Cox regression (statsmodels PHReg or lifelines) on lag predictors.
6. **Benchmark.** Parse FDA quarterly tables (`parse_potential_signals_html`, `load_potential_signals`), map to (ingredient, PT), and compute detection rate, lead time and precision proxy (`evaluate_against_fda`); compare IC vs ROR vs PRR criteria and k_sustain choices.

## Concrete specifications

**Ingredient universe.** Active ingredients (`openfda.substance_name`/`generic_name` normalised, salts stripped) with ≥ 1 current `HUMAN PRESCRIPTION DRUG` SPL and ≥ 1 000 FAERS reports as suspect drug 2004–2025 (~600 ingredients). Reference label = the NDA/BLA set id with the latest `effective_time`; generics pooled as supporting labels.

**Report universe for the scan.** FAERS reports 2004Q1–2025Q2, latest version per `safetyreportid`, suspect drugs only (`drugcharacterization == 1`), all countries (US-only as sensitivity), PTs as reported (MedDRA version drift resolved by PT name; HLT via MedDRA for grouping).

| Object | Definition | Module / function |
|---|---|---|
| Labelled PT set `L(d)` | PTs with ≥ 1 affirmed mention in Boxed Warning, Warnings & Precautions, Warnings, Precautions or Adverse Reactions of any label of ingredient *d*; subsets `L_pm(d)` (postmarketing-only) and `L_bw(d)` (boxed) | `term_extractor.extract_from_label`, `labeled_terms` |
| Cumulative 2x2 at quarter *q* | a = reports of *d* with PT *e* received ≤ q; margins over distinct reports | `signal_scan.cumulative_pair_tables` |
| Sustained signal | IC025 > 0 and a ≥ 3 for k_sustain = 2 consecutive quarters (sensitivity: ROR025 > 1; PRR ≥ 2 & χ² ≥ 4; k = 3) | `signal_scan.time_scan` |
| First signal date `T_s(d,e)` | start of the first quarter of a sustained signal | `time_scan` → `first_signal_quarter` |
| Unlabeled signal | sustained signal at the data cut with `e ∉ L(d)` after exact/synonym/HLT mapping | `signal_scan.flag_unlabeled` |
| First labelled date `T_l(d,e)` | earliest `effective_time` (DailyMed history) or SrLC approval date of a version whose safety sections contain *e* | `label_lag.first_labeled_dates`, `srlc_terms` |
| Lag | `T_l − T_s` in months; categories signal-first / label-first / censored | `label_lag.compute_label_lag` |
| FDA benchmark | (quarter, ingredient, PT) from the quarterly potential-signals tables; detected if `T_s ≤` FDA quarter | `label_lag.load_potential_signals`, `evaluate_against_fda` |

**Output tables (parquet under `data/processed/`).**
- `label_terms`: (set_id, ingredient, version, effective_time, section, pt, n_mentions, n_affirmed, postmarketing_only).
- `pair_scan`: `time_scan` output for every (ingredient, pt) with cumulative n ≥ 3.
- `unlabeled_map`: `rank_unlabeled` output + seriousness share + reporter mix + `post_dsc` share.
- `label_lag`: `compute_label_lag` output with KM inputs.
- `fda_benchmark`: per FDA entry, detected flag and lead quarters.

**Quick start.**
```bash
pip install -r requirements.txt
export OPENFDA_API_KEY=...                                        # optional
python scripts/download_data.py --sample --start-quarter 2018Q1 --end-quarter 2024Q4
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import pandas as pd
from unlabeled_ae import signal_scan as ss, term_extractor as te, label_client as lc
raw = "data/raw/openfda_sample/"
tables = ss.cumulative_from_counts(pd.read_csv(raw+"faers_pair_counts.csv"), pd.read_csv(raw+"faers_drug_totals.csv"),
                                   pd.read_csv(raw+"faers_pt_totals.csv"), pd.read_csv(raw+"faers_all_totals.csv"))
scan = ss.time_scan(tables)
secs = pd.read_csv(raw+"label_sections.csv")
d = te.MedDRADictionary.from_seed()          # replace with from_meddra_ascii(...) for research use
labeled = {}
for (drug, sid), g in secs.groupby(["generic_name", "set_id"]):
    doc = lc.LabelDoc(sid, None, None, None, [drug], [], [], [], None, dict(zip(g.section, g.text.fillna(""))))
    labeled.setdefault(drug.split()[0], set()).update(te.labeled_terms(te.extract_from_label(doc, d)))
print(ss.rank_unlabeled(ss.flag_unlabeled(scan, labeled, dictionary=d)).head(20))
EOF
python scripts/download_data.py --potential-signals
pytest -q tests
```

## Evaluation & statistics

- **Extraction:** precision/recall/F1 at PT level vs manual annotation (two annotators, Cohen's κ), stratified by section; error analysis for table-derived terms.
- **Scan validity:** OMOP/EU-ADR reference sets of positive and negative drug–event controls for AUC of the current-signal statistic; time-scan calibration by checking that negative controls rarely acquire sustained signals (false-emergence rate per 1 000 pair-quarters).
- **Multiplicity:** millions of pair-quarters → report BH-FDR q-values on the current-quarter IC test alongside the conventional IC025 > 0 rule; hierarchy-aware grouping (HLT) to reduce redundant signals.
- **Lag inference:** KM with right-censoring at the data cut; label-first pairs excluded from the at-risk set (immortal-time guard); Cox with drug-level clustering.
- **Benchmark leakage:** FDA lists are not used to tune thresholds; thresholds fixed a priori (pre-registered), with a sensitivity grid reported.
- **Robustness:** suspect-only vs all drugs; US-only reports; exclusion of literature and study reports; alternative term mapping (PT vs HLT).

## Publishable angle

- **Headline:** "Across N US prescription ingredients, X% of sustained FAERS signals are unlabeled; where labels were later updated the median signal-to-label lag was Y months; the open scan would have flagged Z% of FDA's own potential signals a median of W quarters earlier" — plus a public, quarterly-updated unlabeled-signal map.
- **Venues:** *Drug Safety*; *Clinical Pharmacology & Therapeutics*; *JAMIA* or *Journal of Biomedical Informatics* (extraction + benchmark); *BMJ* / *JAMA Internal Medicine* research letter for the label-lag headline; *Pharmacoepidemiology and Drug Safety*.
- **Follow-ups:** LLM-assisted causality triage of the top unlabeled signals; extension to EU SmPCs vs EudraVigilance (label divergence US vs EU); prospective evaluation against future FDA quarterly lists; integration with the sibling bias-adjusted FAERS framework.

## Risks, confounds & mitigations

- **MedDRA licence and mapping errors** (FAERS PTs vs label LLTs, UK/US spelling, class effects) → licensed MedDRA hierarchy, HLT-level matching sensitivity, manual audit of 300 unlabeled flags.
- **Label text heterogeneity** (tables, footnotes, negations, "not established" language) → section-aware parsing, negation window, table parsing of `adverse_reactions_table`, validation corpus.
- **Stimulated reporting after label changes** (label-first pairs) → explicit categorisation (H4), exclusion from lag estimates, ITS sensitivity.
- **Indication and channelling confounds** (events that are disease symptoms) → indication-restricted comparators (other drugs with the same `pharm_class_epc`), and flagging of indication terms extracted from `indications_and_usage`.
- **Incomplete version history** (DailyMed history begins ~2005; openFDA effective_time is the current version only) → SrLC for 2016+, DailyMed archive zips, and interval censoring in the lag model.
- **FDA list heterogeneity** (signals from case series, literature, Sentinel; renamed AEMS in 2026) → restrict benchmark to entries attributed to FAERS review; treat list membership as noisy ground truth.

## Milestones

- [ ] Bulk label pull; extractor on all Rx labels; validation on 100 labels + Demner-Fushman corpus (H5).
- [ ] FAERS bulk ingest (2004–2025), pair tables, time scan; negative-control calibration.
- [ ] Unlabeled-signal map v1 with ranking; audit 300 flags (H1).
- [ ] DailyMed history / SrLC ingestion; label-change dates; lag analysis (H2, H4).
- [ ] FDA potential-signals parsing and benchmark (H3); threshold sensitivity grid.
- [ ] Release map + code; manuscript.

## Ethics / data-use notes

All inputs are public and de-identified; FAERS narratives are not used. MedDRA is licensed — never commit it and do not upload MedDRA content to third-party services. Unlabeled signals are hypotheses, not evidence of causation; the released map must carry that caveat and FDA's standard FAERS disclaimer. Never commit data (`.gitignore`).
