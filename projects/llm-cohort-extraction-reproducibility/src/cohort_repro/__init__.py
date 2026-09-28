"""cohort_repro: paper -> structured cohort definition -> executable MIMIC cohort, locally.

Modules
-------
llm_client   Provider-agnostic chat client with a LOCAL default (OpenAI-compatible local server or
             in-process transformers) and a guard against remote endpoints; JSON helpers.
cohort_spec  Typed cohort-definition DSL (criteria vocabulary, outcome), extraction prompt,
             parser/validator, canonical hashing and multiverse expansion over ambiguities.
compiler     Deterministic DSL -> DuckDB SQL for MIMIC-IV / MIMIC-III, plus a pandas executor with
             identical semantics for the demo databases and tests.
evaluation   Cohort-size / prevalence errors (incl. the Johnson-2017 within-25% flag), criterion-level
             agreement, self-consistency, multiverse summaries and benchmark tables.
"""

from . import cohort_spec, compiler, evaluation, llm_client

__all__ = ["cohort_spec", "compiler", "evaluation", "llm_client"]
__version__ = "0.1.0"
