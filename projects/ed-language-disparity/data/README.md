# Data acquisition

Nothing here is committed. Expected layout:

```
data/
  mimic-iv-ed/ed/        # credentialed: edstays, triage, vitalsign, pyxis, medrecon, diagnosis (csv.gz)
  mimiciv/hosp/          # credentialed: admissions, patients, transfers
  mimiciv/icu/           # credentialed: icustays
  mimic-iv-note/note/    # credentialed: discharge.csv.gz (interpreter regex only)
  mimic-iv-ed-demo/ed/   # open demo for dry runs
  mimic-iv-demo/{hosp,icu}/
  derived/               # visit table, features, model predictions
```

## Credentialed PhysioNet resources

1. PhysioNet account, CITI "Data or Specimens Only Research", sign the DUA for each resource (MIMIC-IV-ED v2.2, MIMIC-IV v3.1, MIMIC-IV-Note v2.2).
2. `export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...`
3. `python scripts/download_data.py --ed --hosp --note` runs `wget -N -c --user ... --password ...` for exactly the tables listed above (MIMIC-IV-Note `discharge.csv.gz` is ~1 GB; `radiology.csv.gz` is not needed here).

Important: use MIMIC-IV **>= 3.0** — the `admissions.language` column was changed in v3.0 from `ENGLISH` / `?` to a standardised primary language. Run `python scripts/download_data.py --language-audit` after download to print the value counts and check the mapping in `src/ed_language/cohort.py::harmonize_language` against them.

## Open demos (no credentials)

`python scripts/download_data.py --sample` fetches the MIMIC-IV-ED Demo v2.2 and MIMIC-IV Demo v2.2 tables so the whole pipeline can be exercised on 100 patients. The demo `language` field follows the version it was built from (v2.2 binary); the audit will show `ENGLISH` / `?` there.

## Benchmark code

`git clone https://github.com/nliulab/mimic4ed-benchmark` (Xie et al., 2022) for the reference cohort and outcome definitions; only the definitions are reused, no data.

## Checks

`python scripts/download_data.py --check` lists which parts are present.
