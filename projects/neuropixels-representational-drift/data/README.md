# Data acquisition

Nothing here is committed except this file. All datasets are open. The
recommended workflow is *streaming* (no bulk download): NWB files are opened
over HTTP with `remfile`/`h5py` and only the units table, spike times and
stimulus tables are cached locally as `.npz`/`.parquet`.

## Expected layout

```
data/
  cache/
    visual_coding/<session_id>.npz          # units, spike times, stimulus tables (from scripts/download_data.py)
    visual_behavior/<session_id>.npz
    ibl/<eid>.npz
  manifests/
    dandi_000021_assets.csv                 # asset path, size, s3 url
    dandi_000022_assets.csv
    ibl_bwm_sessions.csv
  nwb/                                      # optional local NWB copies (large)
```

## 1. Allen Visual Coding - Neuropixels (DANDI 000021 / 000022)

Two dandisets hold the 58 sessions (Brain Observatory 1.1 and Functional
Connectivity stimulus sets). Confirm names/contents with the CLI first:

```bash
pip install dandi pynwb remfile
dandi ls DANDI:000021 | head            # dandiset metadata
dandi ls -r DANDI:000021 | head -50     # assets (one NWB per session plus per-probe LFP files)
```

List assets and stream one session with the script (no files > few MB are written):

```bash
python scripts/download_data.py --dataset visual_coding --list            # writes data/manifests/dandi_00002{1,2}_assets.csv
python scripts/download_data.py --dataset visual_coding --sample          # stream the first session NWB, cache units + stimulus tables
python scripts/download_data.py --dataset visual_coding --session 715093703   # by Allen session id (matched against asset path)
python scripts/download_data.py --dataset visual_coding --download        # full local copies via `dandi download` (~150 GB)
```

Alternative (AllenSDK cache, downloads NWBs from S3):

```python
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest="data/ecephys_cache_dir/manifest.json")
sessions = cache.get_session_table()
session = cache.get_session_data(sessions.index[0])
```

Stimulus tables of interest: `natural_movie_one_presentations` (2 blocks x 10 repeats in Brain Observatory 1.1; 60 repeats in Functional Connectivity as `natural_movie_one_more_repeats`), `natural_movie_three_presentations`, `drifting_gratings_presentations`, `static_gratings_presentations`.

## 2. Allen Visual Behavior - Neuropixels (AWS S3 via AllenSDK)

```bash
pip install allensdk
python - <<'EOF'
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir="data/visual_behavior_neuropixels_cache")
sessions = cache.get_ecephys_session_table()
print(sessions.head())
session = cache.get_ecephys_session(ecephys_session_id=sessions.index[0])
EOF
```

The cache downloads per-session NWBs (~2-4 GB each) on demand from the public
`visual-behavior-neuropixels-data` S3 bucket. `scripts/download_data.py --dataset visual_behavior --sample`
wraps this for one session and caches units/stimulus tables.

## 3. IBL Brain-wide Map (ONE API / AWS Open Data)

```bash
pip install ONE-api ibllib
python - <<'EOF'
from one.api import ONE
one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
eids = one.search(project="brainwide", task_protocol="ephys")   # BWM sessions
print(len(eids))
eid = eids[0]
trials = one.load_object(eid, "trials")
pids, _ = one.eid2pid(eid)
from brainbox.io.one import SpikeSortingLoader
ssl = SpikeSortingLoader(pid=pids[0], one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)   # adds acronyms + quality label
EOF
```

`scripts/download_data.py --dataset ibl --sample` wraps this for the first session.
The public S3 mirror (`s3://ibl-brain-wide-map`, region us-east-1, no credentials)
can be browsed with `aws s3 ls --no-sign-request s3://ibl-brain-wide-map/`.
A DANDI mirror also exists (search dandiarchive.org for "IBL Brain Wide Map").

## 4. Region hierarchy

Area acronyms are mapped to groups in `npx_drift.quality.area_group` (cortex /
thalamus / hippocampal formation / midbrain / other). For a full CCF tree use
AllenSDK `ReferenceSpaceCache` or the `structure_tree` bundled with the ecephys cache.

## Hygiene

- Keep `data/cache` under a few GB by caching binned responses (`.npz`), not spike trains, for pooled analyses.
- Record dandiset versions (`dandi ls DANDI:000021` shows the version tag) and AllenSDK version in `data/manifests/versions.txt`.
