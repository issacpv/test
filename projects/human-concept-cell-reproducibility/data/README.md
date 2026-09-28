# Data acquisition

All data are open NWB files on the DANDI archive. Nothing here needs credentials.
Never commit `.nwb` files (the repository `.gitignore` excludes `data/*`).

## 1. Install the DANDI client

```bash
pip install "dandi>=0.60" pynwb remfile
```

## 2. Enumerate human single-unit dandisets

The project downloader queries the DANDI REST API (`https://api.dandiarchive.org/api/dandisets/?search=...`)
and lists candidate dandisets with their sizes:

```bash
python scripts/download_data.py --list
```

Known starting points:

| Dandiset | Paper | Task | Notes |
|---|---|---|---|
| DANDI:000004 | Chandravadia et al., 2020, Sci Data | New/old recognition memory, 5 image categories | Rutishauser lab; `units` table has waveforms + quality metrics |
| DANDI:000469 | Kyzar et al., 2024, Sci Data | Sternberg working memory (encoding / maintenance / probe) | MTL + medial frontal cortex; 1,809 units, 41 sessions, 21 patients |
| (search "human" "single unit" "object recognition") | Sci Data 2024 object-recognition dataset | Natural images | follow the paper's data-availability statement for the exact identifier |

Verify identifiers on https://dandiarchive.org before a full download; dandiset numbers
are stable but new human datasets appear regularly.

## 3. Download

Full dandiset (resumable, parallel):

```bash
dandi download DANDI:000004 --output-dir data/dandi --jobs 4
dandi download DANDI:000469 --output-dir data/dandi --jobs 4
```

Smoke test (smallest NWB asset of each dandiset, via the REST API, no `dandi` CLI needed):

```bash
python scripts/download_data.py --sample
```

Streaming without a full download (recommended for exploration):

```python
import pynwb, remfile, h5py
url = "https://api.dandiarchive.org/api/assets/<asset_id>/download/"
f = h5py.File(remfile.File(url), "r")
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()
print(nwb.units.colnames, nwb.trials.colnames)
```

Asset ids come from `python scripts/download_data.py --assets 000004`.

## 4. Reference code from the data producers

```bash
git clone https://github.com/rutishauserlab/workingmem-release-NWB  # DANDI:000469 analysis code
```

The recognition-memory release also has a companion repository under the same GitHub
organisation; use it to reproduce the authors' own concept-cell criterion (anchor result).

## Expected layout

```
data/
  README.md                 (this file)
  dandisets.json            (output of --list: id, name, size, asset count)
  dandi/
    000004/
      dandiset.yaml
      sub-P9HMH/sub-P9HMH_ses-20060217_ecephys+image.nwb   (example naming)
      ...
    000469/
      ...
  sessions/                 (derived, produced by src/concept_cells/nwb_loader.py)
    000004/<session_id>.npz  (spike times per unit, trial table, region labels)
  mapping/
    000004.json             (NWB column -> SessionData field mapping)
    000469.json
```

## Size

DANDI:000004 and DANDI:000469 together are on the order of tens of GB (waveform snippets
dominate). Spike times + trial tables extracted into `data/sessions/` are < 1 GB.
