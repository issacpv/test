# Data acquisition

Nothing in this folder is committed. MIMIC is credentialed; papers and models are public.

## 1. MIMIC demo databases (open; pipeline development, CI)

```bash
cd projects/llm-cohort-extraction-reproducibility
python scripts/download_data.py --mimic-demo --out data/mimic-iv-demo       # MIMIC-IV demo v2.2 (100 patients)
python scripts/download_data.py --mimic3-demo --out data/mimic-iii-demo     # MIMIC-III demo v1.4
```

## 2. MIMIC-IV v3.1 and MIMIC-III v1.4 (credentialed)

1. CITI "Data or Specimens Only Research" course; PhysioNet credentialing; sign the DUA per project.
2. Export credentials (never commit them) and run:

```bash
export PHYSIONET_USERNAME=you
export PHYSIONET_PASSWORD=...
python scripts/download_data.py --mimic-full  --out data/mimiciv  --run   # dry-run without --run
python scripts/download_data.py --mimic3-full --out data/mimiciii --run
```

Tables needed by the compiler: `patients`, `admissions`, `icustays`, `diagnoses_icd`, `labevents`
(MIMIC-IV: `hosp/` and `icu/`; MIMIC-III: flat). Load into DuckDB:

```python
import duckdb
con = duckdb.connect("data/mimiciv.duckdb")
for t, sub in [("patients","hosp"),("admissions","hosp"),("diagnoses_icd","hosp"),("labevents","hosp"),("icustays","icu")]:
    con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('data/mimiciv/physionet.org/files/mimiciv/3.1/{sub}/{t}.csv.gz')")
```

Older MIMIC-IV versions (v1.0, v2.0, v2.2) are still on PhysioNet; use the version each paper states.

## 3. Paper corpus (open)

```bash
python scripts/download_data.py --papers --out data/papers            # PubMed candidates + PMC OA full text (XML)
python scripts/download_data.py --papers --sample --out data/papers   # 25 candidates only
python scripts/download_data.py --johnson2017 --out data/papers       # README + study list template for the 2017 calibration set
```

Writes `data/papers/pubmed_candidates.csv`, `data/papers/pmc/PMC*.xml`, and `data/papers/paper_list.csv`
(template with columns `paper_id, doi, pmid, pmcid, database, mimic_version, reported_n,
reported_prevalence, unit, johnson2017_reproduced_n, notes`). Non-OA papers must be obtained through
your library; store their methods sections as `data/papers/text/<paper_id>.txt`.

The Johnson et al. (2017) repository (https://github.com/alistairewj/reproducibility-mimic) contains
the SQL used for their reproductions; clone it next to `data/` to compare cohort sizes.

## 4. Local models (open weights)

```bash
python scripts/download_data.py --models        # prints download + serving commands
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b-instruct
python -m vllm.entrypoints.openai.api_server --model models/llama-3.1-8b-instruct --port 8000 --max-model-len 16384
```

Any OpenAI-compatible local server works (vLLM, llama.cpp `llama-server`, Ollama on port 11434 with
`/v1`, LM Studio). Set `COHORT_REPRO_BASE_URL` and `COHORT_REPRO_MODEL` accordingly. Record model file
hashes in `outputs/model_manifest.json`.

## Expected layout

```
data/
  mimic-iv-demo/physionet.org/files/mimic-iv-demo/2.2/{hosp,icu}/*.csv.gz
  mimiciv/physionet.org/files/mimiciv/3.1/{hosp,icu}/*.csv.gz
  mimiciii/physionet.org/files/mimiciii/1.4/*.csv.gz
  mimiciv.duckdb
  papers/pubmed_candidates.csv
  papers/paper_list.csv
  papers/pmc/*.xml
  papers/text/<paper_id>.txt
  papers/gold/<paper_id>.json          # human-annotated CohortDefinition JSON
models/
outputs/
  definitions/<model>/<paper_id>_seed<k>.json
  cohorts/<model>/<paper_id>.csv
```
