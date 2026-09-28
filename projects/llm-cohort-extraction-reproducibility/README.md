# llm-cohort-extraction-reproducibility

**Can locally-run open-weight LLMs turn the methods sections of published MIMIC studies into executable, auditable cohort definitions, and how much of MIMIC's cohort irreproducibility is due to under-specified criteria versus extraction error? A benchmark against the hand-reproduced cohorts of Johnson et al. (2017) plus a multiverse over ambiguous criteria.**

## Status / difficulty / timeline / compute

- Status: design + starter code (provider-agnostic LLM client with a local-inference default and a remote-URL guard; a typed cohort-definition DSL with an extraction prompt and parser; a deterministic DSL-to-SQL compiler for MIMIC-IV/III with a pandas executor for the demo database; evaluation metrics incl. the Johnson-2017 "within 25%" criterion, criterion-level F1, self-consistency and multiverse spread). No data or models shipped.
- Difficulty: MSc-level (NLP + clinical-data engineering); the hard part is annotation and the compiler's coverage of MIMIC idioms, not the models. 6-9 months.
- Compute: one GPU with 24-48 GB VRAM (or 2x24 GB) for 7B-32B open-weight models via vLLM/llama.cpp; CPU-only is possible with quantised 7-8B models. DuckDB over MIMIC-IV on a 64 GB workstation.

## Background

MIMIC-III/IV underpin thousands of papers, and a large fraction of them define a cohort in prose: "adult patients, first ICU stay, length of stay >= 24 h, excluding those who died within the first 6 hours". Johnson, Pollard & Mark (2017, MLHC) tried to reproduce the cohorts of 28 mortality-prediction papers (38 experiments) from those descriptions and found that in half of the experiments their cohort size differed from the reported one by more than 25%, with the largest discrepancy above 11,000 patients. The MIMIC Code Repository (Johnson et al., 2018, JAMIA), standard benchmarks (Harutyunyan et al., 2019, Sci Data; Purushotham et al., 2018, J Biomed Inform) and pipelines (Gupta et al., 2022, MLHC for MIMIC-IV) exist precisely to make cohorts reproducible, but most papers still do not use them, and the prose-to-cohort step is done by hand.

Large language models are now routinely used for adjacent tasks: parsing eligibility criteria into queries (Criteria2Query, Yuan et al., 2019, JAMIA; TrialGPT, Jin et al., 2024, Nat Commun), text-to-SQL over MIMIC (EHRSQL, Lee et al., 2022, NeurIPS Datasets & Benchmarks), automated cohort construction from free-text criteria on EHR databases ("Leveraging foundation language models for automated cohort extraction from large EHR databases", arXiv 2024), and finding cohort *names* in the literature (arXiv 2026). The unaddressed question is the reproducibility one: given a *published paper*, can an LLM produce a structured, executable cohort definition whose realised cohort matches what the authors report, and when it does not, is the failure the model's or the paper's?

A second constraint shapes the design. PhysioNet's policy on the use of MIMIC with LLMs ("Use of MIMIC Data with Large Language Models and Online Services") prohibits sending credentialed data to third-party services that do not meet its retention/training requirements and recommends locally deployed models. This project therefore uses only locally-run open-weight models, and, more strongly, never lets the model see MIMIC records at all: the model reads *public papers* and emits a definition; a deterministic compiler executes it locally.

## The research gap

What has been done:

- Manual reproducibility audit of MIMIC cohorts (Johnson et al., 2017): 28 papers, human effort, MIMIC-III only, no analysis of *which* criteria drive the discrepancies.
- Text-to-SQL on MIMIC (EHRSQL and successors): questions are short, synthetic, and evaluated on execution accuracy against gold SQL, not on reproducing published cohorts.
- Eligibility-criteria parsing for trials (Criteria2Query, TrialGPT): trial criteria, OMOP/trial registries, not observational cohort definitions in papers and not MIMIC.
- Automated cohort extraction with foundation models (arXiv 2024): free-text criteria supplied by a user, not extracted from papers; evaluated on a single database with API models.
- Reviews of LLMs in clinical research warn about API-based evaluation on protected data (Wornow et al., 2023, npj Digit Med) but do not benchmark local models on this task.

What is specifically missing (our angle):

1. **Paper -> structured definition -> executable cohort**, with the definition in a small, typed DSL (`cohort_spec.py`) rather than free SQL, so that every criterion is auditable and *ambiguities are explicit* (each criterion carries `ambiguous` and `alternatives`).
2. **Ground truth at two levels**: (a) human-annotated structured definitions for a gold subset (criterion-level agreement), and (b) realised cohort sizes / outcome prevalences compared with the paper's reported numbers and with Johnson et al.'s hand-reproduced numbers.
3. **Attribution of irreproducibility**: a multiverse over the alternatives the model flags as ambiguous gives the *range* of cohort sizes consistent with the paper; a reported n outside that range indicates an unreported step, inside it indicates under-specification. Nobody has quantified this split.
4. **Open-weight, local models only**, across sizes (7B to 70B, quantised and full precision), with self-consistency across seeds, so results are reproducible by any credentialed group without API access.
5. **A deterministic compiler vs LLM-written SQL** comparison on the same papers: does forcing the model through a DSL reduce execution errors and silent cohort drift?

## Research questions / hypotheses

1. **RQ1 (reproduction rate).** For N ~ 60 MIMIC-IV papers (2020-2025) and the 28 MIMIC-III papers of Johnson et al., what fraction of LLM-extracted, compiled cohorts fall within 25% of the reported n? H1: the best local model (>= 30B) achieves >= 60% within-25%, matching or exceeding the human rate reported by Johnson et al. (50%), because the compiler applies the same conventions consistently.
2. **RQ2 (where the errors are).** H2: after multiverse expansion, >= 70% of reported cohort sizes fall inside the range spanned by the flagged alternatives; the criteria most often responsible are (a) the definition of "first ICU stay" (per subject vs per hospital admission), (b) minimum length-of-stay thresholds and whether they apply to ICU or hospital stay, and (c) age computation for patients > 89.
3. **RQ3 (DSL vs free SQL).** H3: the DSL route has fewer execution failures (< 5% vs > 20%) and smaller median absolute relative error in cohort size than model-written SQL from the same paper text.
4. **RQ4 (model size and quantisation).** H4: criterion-level F1 against the gold subset rises with model size and plateaus at ~30B; 4-bit quantisation costs < 3 F1 points; self-consistency across 5 seeds exceeds 0.8 Jaccard for the best models.
5. **RQ5 (downstream impact).** H5: replacing the reported cohort with the reconstructed cohort changes the reported outcome prevalence by more than 2 percentage points in >= 30% of papers, enough to shift AUROC of a re-trained baseline model by > 0.01.

## Datasets

| Dataset | Used for | Size | Access | URL |
|---|---|---|---|---|
| MIMIC-IV v3.1 (hosp, icu) | executing compiled cohorts for MIMIC-IV papers | ~94k ICU stays; ~30 GB csv.gz | PhysioNet credentialed (CITI + DUA) | https://physionet.org/content/mimiciv/3.1/ |
| MIMIC-III v1.4 | executing cohorts for the Johnson et al. (2017) calibration set | ~61k ICU stays; ~6 GB | PhysioNet credentialed | https://physionet.org/content/mimiciii/1.4/ |
| MIMIC-IV demo v2.2 / MIMIC-III demo v1.4 | pipeline development and CI (pandas executor) | 100 patients | Open | https://physionet.org/content/mimic-iv-demo/2.2/ |
| Johnson et al. 2017 reproducibility repository | hand-reproduced cohort sizes and SQL for 28 MIMIC-III papers (calibration ground truth) | code + tables | Open (GitHub) | https://github.com/alistairewj/reproducibility-mimic |
| MIMIC Code Repository | reference concept SQL (first ICU stay, age, SOFA, etc.) used by the compiler | code | Open | https://github.com/MIT-LCP/mimic-code |
| Paper corpus (built here) | 60 MIMIC-IV papers via PubMed/PMC E-utilities (OA full text where available) | text | Open (papers); non-OA via library | https://eutils.ncbi.nlm.nih.gov |
| Open-weight instruction models (e.g. Llama 3.x, Qwen 2.5/3, Mistral, Gemma families) | extraction | 7B-70B weights | Open weights under each model's licence (Hugging Face) | https://huggingface.co |

Paper selection rule: PubMed search `MIMIC-IV[tiab] AND (mortality OR readmission OR "length of stay") AND (cohort OR patients)`, 2020-2025, screened for (i) a single primary cohort defined in prose, (ii) a reported cohort size, (iii) an ICU-stay or admission unit of analysis. Stratify by journal type (clinical vs ML venue).

## Methods

Pipeline (modules in `src/cohort_repro/`):

1. **LLM access** (`llm_client.py`). A `LLMClient` protocol with three implementations: `OpenAICompatibleClient` (default `http://localhost:8000/v1`, i.e. vLLM / llama.cpp server / Ollama / LM Studio), `TransformersClient` (in-process Hugging Face generation), and `DryRunClient` (canned responses for tests). Non-local base URLs are refused unless `COHORT_REPRO_ALLOW_REMOTE=1` *and* `allow_remote=True` are both set, and even then the pipeline only ever sends paper text, never MIMIC rows. JSON-mode helper with schema validation and bounded repair attempts.
2. **Cohort DSL and extraction** (`cohort_spec.py`). `CohortDefinition` = database + unit of analysis (ICU stay / hospital admission / patient) + prediction time + list of typed `Criterion` objects from a closed vocabulary (age bounds, first-stay rules, LOS thresholds, care-unit filters, ICD-prefix inclusion/exclusion, required measurements, early-death exclusions, ...) + `Outcome`. Each criterion carries the quoted source sentence, an `ambiguous` flag and `alternatives`. The extraction prompt enumerates the vocabulary and asks for JSON; `parse_extraction` validates and normalises; `enumerate_multiverse` expands alternatives.
3. **Compiler and executors** (`compiler.py`). `compile_sql` turns a definition into DuckDB SQL over the MIMIC-IV (or MIMIC-III) tables using MIMIC-Code conventions (age from `anchor_age`/`anchor_year`, `> 89` handling, first stay by `intime` within subject or admission). `PandasExecutor` applies the identical semantics to in-memory tables so the demo database and unit tests run without DuckDB; SQL and pandas paths are cross-checked.
4. **Baselines.** (a) LLM-written SQL from the same paper text and a schema card (the text-to-SQL route); (b) rule-based regex extraction of age/LOS thresholds; (c) human-annotated definitions on the gold subset (two annotators, adjudicated).
5. **Evaluation** (`evaluation.py`). Cohort-size error (absolute relative error, log ratio, within-25% flag), prevalence error, criterion-level precision/recall/F1 against gold, self-consistency across seeds, multiverse range coverage, execution-failure rate.
6. **Downstream impact.** For papers with a mortality outcome, re-train a logistic baseline on the reconstructed vs reported-convention cohort and compare AUROC and prevalence.

Models: at least one model each from the 7-9B, 14-32B and 70B classes, full precision and 4-bit; temperature 0 for the main run, 5 seeds at temperature 0.7 for self-consistency.

Tools: vLLM or llama.cpp (server), transformers, DuckDB, pandas, MIMIC-Code SQL concepts, PubMed E-utilities.

## Evaluation & statistics

- Unit of analysis: one paper (primary cohort). Calibration set: the Johnson et al. (2017) papers with their hand-reproduced n as an additional reference.
- Primary metric: fraction within 25% of reported n (as in Johnson et al.) with Wilson CIs; secondary: median absolute relative error, criterion F1 on the gold subset (n = 20 papers, two annotators, Cohen's kappa reported).
- Model comparisons: paired by paper; McNemar for within-25% flags, Wilcoxon for errors; Holm across the five pre-registered hypotheses.
- Leakage: the extraction prompt contains only the paper text and the DSL; no cohort counts are fed back to the model. Repair loops only see JSON-validation errors, never database results.
- Determinism: fixed seeds, pinned model files (SHA-256 recorded), greedy decoding for the main run; every definition is stored as canonical JSON with its hash.
- Nulls: (i) a scrambled-paper control (methods text from a different paper) must yield definitions that do not reproduce the target n; (ii) a "criteria-removed" ablation quantifies each criterion's marginal effect on n.

## Publishable angle

Headline: "Open-weight LLMs run entirely on-premise reconstruct the cohorts of X% of published MIMIC studies within 25% of the reported size, better than the manual rate of 50% reported in 2017; most residual discrepancies fall inside the range implied by ambiguities in the papers themselves, and three recurrent under-specifications (first-stay rule, LOS threshold target, age > 89) explain most of it."

Target venues: JAMIA; npj Digital Medicine; Journal of Biomedical Informatics; Machine Learning for Healthcare (MLHC) or CHIL; a reporting-guideline note in Critical Care Medicine / Intensive Care Medicine.

Follow-ups: extend the DSL to eICU and AmsterdamUMCdb (cross-database reproduction); a reporting checklist ("MIMIC cohort card") generated from the DSL; automated screening of new preprints for under-specified cohorts.

## Related projects

`icu-model-transportability` builds cohorts on four ICU databases with a verified ontology; this project's DSL could later target its ontology. `ed-triage-mimic-ed` and `ventilation-policy-offline-rl` in this repository also define MIMIC cohorts and are natural test cases. No code is shared.

## Risks, confounds & mitigations

- **Compiler coverage.** Papers use concepts the DSL lacks (SOFA at 24 h, sepsis-3 onset). Mitigation: track "unsupported criterion" explicitly; add concepts from MIMIC-Code incrementally; report coverage as a result.
- **Papers use older MIMIC versions.** MIMIC-IV v1.0 vs v3.1 differ in size. Mitigation: record the version stated; execute on the matching version where available (PhysioNet keeps old versions); otherwise flag.
- **Model licence and reproducibility.** Mitigation: pin model files and record hashes; prefer models with permissive licences.
- **Annotator disagreement on ambiguous criteria.** Mitigation: adjudication protocol; report kappa; ambiguity itself is an outcome.
- **Prompt sensitivity.** Mitigation: three prompt variants; report the spread.
- **Temptation to use API models for comparison.** Not permitted for MIMIC data, and unnecessary here because the model never sees MIMIC; even so, we keep the pipeline local-only to avoid any policy ambiguity and to guarantee reproducibility.

## Milestones

- [ ] Credentialing; MIMIC-IV/III + demo databases loaded in DuckDB; MIMIC-Code concepts built.
- [ ] Paper corpus (60 MIMIC-IV + 28 Johnson-2017 papers); OA full text; gold annotations for 20 papers.
- [ ] DSL + compiler covering the criteria in the gold subset; SQL/pandas cross-check on the demo database.
- [ ] Local model serving (vLLM); extraction runs for 3 model sizes x 2 precisions x 5 seeds.
- [ ] Evaluation tables; multiverse attribution; DSL-vs-SQL baseline.
- [ ] Downstream-impact experiment; pre-registered analysis; manuscript; release of DSL, prompts and definitions (not data).

## Ethics / data-use notes

- MIMIC-III/IV are credentialed (CITI + DUA). Data stay on approved machines; nothing from the database is committed, logged in prompts, or sent to any external service.
- PhysioNet's LLM policy: locally deployed models are the recommended route; this project uses only local open-weight models, and the models never receive MIMIC records, only public paper text and the DSL. `llm_client.py` enforces a local-URL default and refuses remote endpoints unless explicitly overridden, which this project does not do.
- Papers are public; quoting their methods sections in prompts is fair use for research. Extracted definitions and per-paper results will be released; authors of audited papers will be contacted before publication with their reconstructed definitions.
- Model weights are subject to their own licences; record and respect them.
