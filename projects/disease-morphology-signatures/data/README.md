# Data acquisition: disease-morphology-signatures

Everything is open access; nothing in `data/` is committed.

## 1. NeuroMorpho.org REST API v1

Base URL `https://neuromorpho.org/api` (reference: https://neuromorpho.org/apiReference.html).

| Endpoint | Purpose |
|---|---|
| `POST /neuron/select?page=P&size=S` with JSON body such as `{"species": ["mouse"]}` | filtered listing, `size` <= 500, `page` from 0 |
| `GET /neuron/fields/experiment_condition` | the full controlled/free-text vocabulary of experimental conditions (inspect it before editing the keyword table in `src/disease_morph/cohort.py`) |
| `GET /neuron/name/{neuron_name}` | single record |
| `GET /morphometry/name/{neuron_name}` | NeuroMorpho's precomputed L-Measure summary |
| `https://neuromorpho.org/dableFiles/{archive_lowercase}/CNG version/{neuron_name}.CNG.swc` | standardised SWC (URL-encode the space) |

Responses are Spring-HATEOAS style: `{"_embedded": {"neuronResources": [...]}, "page": {...}}`.

Fields used here: `neuron_name`, `archive`, `species`, `brain_region` (list), `cell_type` (list), `experiment_condition` (list), `min_age`, `max_age`, `age_classification`, `gender`, `strain`, `reconstruction_software`, `shrinkage_corrected`, `shrinkage_reported`, `slicing_thickness`, `stain`, `protocol`, `physical_Integrity`, `structural_domains`, `reference_pmid`, `reference_doi`.

### Commands

```bash
cd projects/disease-morphology-signatures
pip install -r requirements.txt

# smoke test: two pages of mouse records, keyword-filter, 5 SWC files
python scripts/download_data.py --sample

# full harvest of the four species that carry almost all disease/aging data
python scripts/download_data.py --species mouse rat human monkey

# print the archive x condition table (after harvesting), no download
python scripts/download_data.py --report

# fetch SWC files for cases + same-archive controls
python scripts/download_data.py --species mouse rat human monkey --swc --workers 4
```

The script is resumable (pages already in the JSONL are skipped; existing SWC files are not re-downloaded). If NeuroMorpho's TLS chain is misconfigured on the day you run this, set `NEUROMORPHO_VERIFY_SSL=0` for the session only.

## 2. Alzforum research-model covariates

Manually build `data/ad_models.csv` with columns `model, transgenes, promoter, plaque_onset_months, tangle_onset_months, source_url` from https://www.alzforum.org/research-models (one row per AD model that appears in the NeuroMorpho condition strings, e.g. APP/PS1, 3xTg-AD, 5xFAD, Tg2576, PS19).

## 3. ModelDB passive parameters

Search https://modeldb.science for the original papers (e.g. "Coskren 2015", "Tejada 2014") and record `Rm`, `Ra`, `Cm` per cell class in `data/passive_params.csv` with columns `cell_class, rm_ohm_cm2, ra_ohm_cm, cm_uf_cm2, source`. Defaults in code: 20,000 Ohm cm^2, 150 Ohm cm, 1 uF/cm^2.

## 4. Optional: Allen Cell Types reference reconstructions

```bash
pip install allensdk
python - <<'EOF'
from allensdk.core.cell_types_cache import CellTypesCache
ctc = CellTypesCache(manifest_file='data/allen/manifest.json')
cells = ctc.get_cells(require_reconstruction=True)
for c in cells[:5]:
    ctc.get_reconstruction(c['id'])  # SWC saved under data/allen/
EOF
```

## Expected layout

```
data/
  README.md
  metadata/
    neurons_mouse.jsonl, neurons_rat.jsonl, ...
    neurons.parquet            # flattened table
    cohort.parquet             # condition-mapped, archive-matched table
    condition_mapping_audit.csv
  swc/<archive>/<neuron_name>.CNG.swc
  ad_models.csv
  passive_params.csv
  allen/                        # optional
```
