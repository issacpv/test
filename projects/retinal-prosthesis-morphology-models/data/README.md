# Data acquisition

Everything is open. Only this README is committed.

## 1. NeuroMorpho.org retinal ganglion cells (REST API v1)

```bash
python scripts/download_data.py --sample                    # 3 species, 25 records each, 10 SWC files
python scripts/download_data.py --metadata --swc            # all RGC records + SWC files
python scripts/download_data.py --metadata --swc --species mouse rat
```

API pattern (https://neuromorpho.org/apiReference.html):

- `GET /api/neuron/select?q=brain_region:retina&q=species:mouse&page=0&size=500`
  (records are filtered client-side to `cell_type` containing "ganglion")
- `GET /api/neuron/fields/cell_type` to see the exact vocabulary
- SWC: `https://neuromorpho.org/dableFiles/<archive lower-case>/CNG version/<neuron_name>.CNG.swc`

Output: `data/neuromorpho/meta/<species>.jsonl`, `data/neuromorpho/swc/<neuron_name>.CNG.swc`.
Set `NEUROMORPHO_VERIFY_SSL=0` only if the site's TLS chain is temporarily broken.

## 2. Eyewire Museum (Bae et al., 2018)

https://museum.eyewire.org -> each cell page offers a download (SWC/OBJ skeleton) and the type
label (47 types), stratification profile and soma position. Save SWC files under
`data/eyewire/<cell_id>.swc` and the cell table (cell id, type, soma x/y/z) as
`data/eyewire/cells.csv`. The site's terms allow research use with citation.

## 3. Sümbül et al. (2014) reconstructions

Deposited to NeuroMorpho (search archive names with `GET /api/neuron/fields/archive` and filter
for "Sumbul"/"Seung"/"Sanes"); they arrive through step 1. Genetic-line labels are in the
`cell_type` field.

## 4. Empirical threshold distributions

- Grosberg et al. (2017) J Neurophysiol 118:1457 (primate; ON/OFF parasol & midget thresholds).
- Madugula et al. (2022) J Neural Eng 19:066040 (human ex vivo).

Transcribe published summary statistics into `data/empirical/thresholds.csv`
(`study, species, cell_type, n, median_uA, iqr_lo, iqr_hi, pulse_us, electrode_um`) and request
raw per-cell tables from the authors where possible.

## 5. Optional: pulse2percept axon map

```bash
pip install pulse2percept
python -c "import pulse2percept as p2p; print(p2p.__version__)"
```

`rgc_prosthesis.swc_morph.synthesize_axon` accepts an axon direction; use
`p2p.models.AxonMapModel` to obtain the local nerve-fibre direction for a soma position.

## Expected layout

```
data/
  README.md
  neuromorpho/meta/*.jsonl
  neuromorpho/swc/*.CNG.swc
  eyewire/*.swc, cells.csv
  empirical/thresholds.csv
outputs/
  morphometrics.parquet
  thresholds_<setting>.parquet
  determinants_model.csv
  selectivity_auc.csv
```
