# device-recall-prediction — predicate-chain risk and MAUDE early warning for FDA device recalls

**Pitch.** Build a device-level longitudinal dataset from openFDA (510(k)/PMA → MAUDE reports → recall class) plus 510(k) predicate chains, and test whether predicate-chain structure (depth, recalled ancestry) and early MAUDE narrative signals predict time to Class I/II recall — with cardiovascular, orthopaedic, insulin-pump and AI-enabled (SaMD) devices as pre-specified subgroups.

**Status:** starter code + protocol. **Difficulty:** MSc–PhD; **timeline:** 9–12 months (3 of them for predicate extraction). **Compute:** CPU workstation; MAUDE bulk (~20 GB JSON) needs ~100 GB disk; sentence-transformer embeddings of ~1 M narratives are optional and take a few GPU-hours.

## Background

Most moderate-risk devices reach the US market through the 510(k) pathway by claiming substantial equivalence to a predicate; a device may cite predicates that were themselves cleared on predicates, producing multi-generation chains ("predicate creep"). Post-market safety relies on MAUDE, a passive system receiving ~1 M reports a year, and on recalls classified by CDRH (Class I = reasonable probability of serious harm). Evidence has accumulated that regulatory characteristics predict recalls (Zuckerman, Brown & Nissen, 2011, *Arch Intern Med*; Everhart et al., 2023, *JAMA*), that recalled predicates propagate risk (Kadakia et al., 2023, *JAMA*: among 156 Class I-recalled 510(k) devices from 2017–2021, 44.1% cited predicates that had themselves been Class I recalled), and that AI-enabled devices are recalled early and often without clinical validation (Lee et al., 2025, *JAMA Health Forum*: 60 of 950 AI devices had 182 recall events, ~43% within a year of authorisation). Cardiovascular devices contribute about a third of all Class I recalls (*Ann Intern Med*, 2024, 137 Class I events 2013–2022).

## The research gap

**Done.**
- *Regulatory predictors of recall:* Everhart et al. (2023, *JAMA*) — submission characteristics; Kadakia et al. (2023, *JAMA*) — one-generation recalled-predicate exposure, Class I only; a 2023 *PLoS ONE* case study of predicate creep in one device lineage.
- *Predicate-network machine learning:* Zhu, Sen, Everhart & Karaca-Mandic (2024, *Information Systems Research*, "DeepPredicate") learn recall risk from the citation network of 45 398 devices (2003–2020) — strong performance but a black-box at clearance time, no MAUDE inputs, and no interpretable estimate of *how* depth or recalled ancestry change hazard. A 2026 *Management Science* study proposes human-plus-algorithm triage of 510(k) submissions.
- *MAUDE text mining:* device-specific NLP (cochlear implants, mesh, hearing aids, digital therapeutics), one small ML recall predictor (*West J Emerg Med*, 2025), and RecallRisk-BERT (Atalay & Yigit-Sert, 2026, arXiv:2606.27174), which classifies recall class/root cause from *recall* narratives, i.e. after the recall exists.
- *AI/ML devices:* Muehlematter et al. (2023, *Lancet Digit Health*) mapped 510(k) predicate networks of AI/ML devices descriptively; Lee et al. (2025) analysed recalls; an *npj Digit Med* (2025) framework paper found MAUDE reports for AI/ML devices largely uninformative.

**Missing — and the contribution of this project.**
1. **Interpretable predicate-chain hazard estimates** across all product codes: does hazard rise with predicate depth (generations to a root device), with the number of recalled ancestors, and with *time-anchored* exposure (ancestor recalled *before* the child's clearance, when the FDA and applicant could have known), using Cox models with product-code strata rather than deep networks?
2. **Early-warning from MAUDE with a landmark survival design**: features from the first 6/12 months of reports (counts, event mix, product-problem diversity, narrative TF-IDF / embeddings) predicting *subsequent* Class I/II recall, with temporal validation — not a post-hoc classification of already-recalled devices.
3. **Joint model**: are predicate-chain and MAUDE-text signals complementary (does text add C-index beyond structure, and vice versa)?
4. **Cross-linked subgroups**: cardiovascular, orthopaedic implants, insulin pumps/CGM, and the FDA AI-enabled device list joined to MAUDE and recalls — the first time all four are analysed with a common device-level dataset and the same estimand.

## Research questions / hypotheses

1. **H1 (depth).** Adjusted for product code (strata), clearance year, clearance type and applicant size, each additional predicate generation increases the hazard of Class I/II recall (HR per generation > 1.05); the association is stronger for implants and life-sustaining devices.
2. **H2 (recalled ancestry, time-anchored).** Devices with ≥ 1 ancestor recalled *before* their clearance have HR ≥ 1.5 for Class I recall vs. devices with no recalled ancestors; ancestry recalled *after* clearance carries a smaller but non-null association (shared design), separating "FDA could have known" from "shared failure mode".
3. **H3 (MAUDE early warning).** A Cox model on 12-month landmark MAUDE features attains temporal-validation C-index ≥ 0.70 for Class I/II recall in the following 5 years; narrative text adds ≥ 0.03 C over counts alone; the most predictive terms concern specific failure modes (e.g. occlusion, fracture, software).
4. **H4 (complementarity).** Combining predicate-chain and MAUDE features improves C-index over either alone, and the improvement is largest for devices with few early reports (structure substitutes for missing surveillance data).
5. **H5 (AI-enabled devices).** AI-enabled devices have shallower predicate chains but higher early recall hazard (reproducing Lee et al. within our framework), and their MAUDE reports are sparser and less specific (lower problem-code diversity) than non-AI devices of the same product code.

## Datasets

| Name | Used for | Size | Access | URL |
|---|---|---|---|---|
| openFDA `device/510k` | cohort of cleared devices (K number, product code, decision date, applicant, clearance type) | ~170 k records (1976–2025) | open | https://open.fda.gov/apis/device/510k/ |
| openFDA `device/pma` | PMA cohort and supplements | ~50 k | open | https://open.fda.gov/apis/device/pma/ |
| openFDA `device/recall` + `device/enforcement` | recall events, linked K/PMA numbers; Class I/II/III via `event_id` join | ~60 k recalls | open | https://open.fda.gov/apis/device/recall/ |
| openFDA `device/event` (MAUDE) | early-warning features: counts, event type, product problems, narratives | > 15 M reports; ~20 GB zipped JSON | open | https://open.fda.gov/apis/device/event/ |
| openFDA `device/classification`, `device/udi` (GUDID) | product-code metadata (class, implant, life-sustaining); UDI → submission linkage | ~7 k codes; ~4 M UDI records | open | https://open.fda.gov/apis/device/ |
| 510(k) summary PDFs (accessdata.fda.gov) | predicate extraction | ~1 PDF per summary submission; 40–100 k PDFs | open (scrape politely) | https://www.accessdata.fda.gov/cdrh_docs/ |
| FDA AI-enabled medical devices list | AI/SaMD subgroup | ~1 500 entries (2025–26) | open (CSV/XLSX) | https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices |
| FDA TPLC database (optional) | per-product-code MDR/recall cross-checks | — | open | https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfTPLC/tplc.cfm |

Acquisition steps and layout: `data/README.md`; `python scripts/download_data.py --sample`.

## Methods

1. **Cohort.** All 510(k) clearances 2003–2020 (follow-up to 2025) and PMAs; exclude 510(k)-exempt product codes. Unit = submission; secondary unit = (product code, firm) family for MAUDE reports that cannot be linked to a submission.
2. **Recall outcome.** `device/recall` × `device/enforcement` join (`openfda_device.join_recall_class`) → first Class I/II recall date per submission via `k_numbers`/`pma_numbers`; fallback linkage via product code + recalling firm (`entity_linking`). Sensitivity: Class I only; any class.
3. **Predicate chains.** Download summary PDFs (`scripts/download_data.py --summaries`), extract K/DEN/P numbers with contextual cues (`predicate_graph.extract_predicates`), build the DAG, remove impossible edges (predicate cleared after child), compute depth, ancestor counts, recalled-ancestor exposure (time-anchored) and PageRank (`graph_features`, `ancestor_recall_exposure`). Manual validation of 300 randomly sampled edges (target precision ≥ 0.95); compare with DeepPredicate's published descriptive statistics.
4. **MAUDE linkage and features.** Flatten MAUDE (`flatten_maude`), de-duplicate, link to submissions via UDI-DI → GUDID `premarket_submissions`, else blocked fuzzy firm/brand match within product code (`link_maude_to_510k`). Landmark features at 6 and 12 months (`survival_text.maude_landmark_features`): counts, deaths/injuries/malfunctions, distinct product-problem codes, within-window slope, narrative TF-IDF (1–2 grams → 20 SVD components, fitted on training years only) and optional MiniLM/PubMedBERT embeddings.
5. **Models.** Cox PH stratified by product code with clearance-year splines (`fit_cox`, lifelines; statsmodels fallback), Fine–Gray competing risk for Class I vs II (lifelines/`cmprsk` in R), and gradient-boosted survival (scikit-survival) as a non-linear benchmark. Baselines: DeepPredicate-style network-only model re-implemented with node2vec + logistic regression; counts-only MAUDE model; Everhart-style regulatory-covariates model.
6. **Subgroups.** Cardiovascular (panel CV), orthopaedic (OR), insulin pumps/CGM (product codes discovered from `device/classification`, e.g. insulin infusion pumps), AI-enabled (FDA list join); interaction tests for H1/H2 by subgroup.

## Evaluation & statistics

- **Temporal validation:** train on clearances 2003–2014, validate 2015–2017, test 2018–2020 with follow-up to 2025 (`temporal_split_evaluate`); no device or narrative from the test years enters TF-IDF/SVD fitting.
- **Metrics:** Harrell's C and time-dependent AUC at 24/60 months; calibration of 5-year recall risk (deciles); precision at top-5% risk ("watch-list" yield); for Class I, sensitivity at fixed specificity 0.95.
- **Inference:** robust (cluster-by-applicant) SEs; Schoenfeld tests for PH; BH-FDR for subgroup and term-level tests; pre-registered primary hypotheses (H1–H3).
- **Nulls / negative controls:** permutation of predicate edges within product code (destroys chain structure but keeps degree); placebo landmark windows *after* recall for text features (should not predict); random text.
- **Leakage guards:** landmark design (features strictly before time origin), recall-date ≥ landmark, no use of recall narratives, applicant-level cluster splits as a sensitivity analysis.
- **Missingness:** summaries unavailable for "Statement" submissions (~20–30%) → inverse-probability weighting on availability; MAUDE-unlinkable reports → family-level analysis.

## Publishable angle

- **Headline:** "Predicate depth and time-anchored recalled ancestry independently increase recall hazard (HR per generation ≈ x; HR ≈ y for pre-clearance recalled ancestors), and 12-month MAUDE narratives add z C-index points, enabling a watch-list that would have flagged w% of Class I recalls ≥ 2 years early — with AI-enabled devices showing shallow chains but the highest early hazard."
- **Venues:** *JAMA Network Open* or *JAMA Internal Medicine* (regulatory science); *npj Digital Medicine* (AI-device subgroup + NLP); *Journal of Biomedical Informatics* / *JAMIA* (methods); *Annals of Biomedical Engineering* or *IEEE JBHI* for the BME framing; RAPS/FDA regulatory-science workshops.
- **Follow-ups:** counterfactual "predicate selection" audit (would FDA's 2023 draft guidance on predicate choice have excluded recalled predicates?); causal-forest heterogeneity by product area; live dashboard using openFDA weekly updates; extension to EUDAMED once vigilance data are public.

## Risks, confounds & mitigations

- **Predicate extraction errors** (OCR, reference vs. predicate devices, numbers cited as comparators) → contextual cues, manual validation set, sensitivity excluding reference-only edges.
- **Surveillance bias**: firms with more reports are not necessarily worse; MAUDE volume scales with sales → adjust for firm size (number of clearances, GUDID device count) and product-code base rates; interpret as early warning, not causal.
- **Recall linkage incompleteness** (`k_numbers` missing for many recalls) → fuzzy firm/product linkage with validation; family-level outcome as fallback.
- **Immortal time / reverse causation** → landmark design; exclude devices recalled before the landmark; report their count.
- **Secular changes** (MDR reporting rules 2019–2021 summary reporting changes, UDI phase-in) → clearance-year splines, calendar-period strata.
- **Class heterogeneity of "recall"** (labelling-only Class II recalls) → sensitivity restricting to design/software root causes (`root_cause_description`).
- **AI-list heterogeneity** (list includes legacy CAD devices) → subgroup by decision year and panel.

## Milestones

- [ ] Bulk download 510k/pma/recall/enforcement/classification/udi; build submission table with first recall date and class.
- [ ] Summary-PDF pipeline for 2003–2020 summaries; predicate DAG; 300-edge validation; depth/ancestry features.
- [ ] MAUDE bulk ingest, de-duplication, UDI/fuzzy linkage; linkage precision audit (200 reports).
- [ ] Landmark feature tables (6/12 months) incl. TF-IDF; Cox + competing-risk models; temporal validation (H1–H4).
- [ ] Subgroup analyses (CV, ortho, insulin pump/CGM, AI-enabled); negative controls; calibration.
- [ ] Watch-list retrospective yield; manuscript; release predicate-edge table and code.

## Ethics / data-use notes

All sources are public FDA data. MAUDE narratives are de-identified but may contain incidental details; do not attempt re-identification and do not send narratives to external LLM APIs without institutional approval (local models are fine). The project produces device-level risk estimates that could be misread as safety verdicts; report as surveillance prioritisation with uncertainty, not as evidence of harm. Never commit data (`.gitignore`).
