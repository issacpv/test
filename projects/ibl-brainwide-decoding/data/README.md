# Data acquisition

All data are open. Nothing under `data/` is committed. Target layout:

```
data/
  ibl/
    bwm_sessions.csv                # from brainwidemap.bwm_query: eid, pid, subject, lab, probe_name, ...
    one_cache/                      # ONE cache (set ONE cache_dir here) with alyx/… per-session datasets
    cache/<eid>/<pid>/<region>.npz  # our binned tensors (produced by ibl_bwm.binning)
  allen/
    ecephys_cache/manifest.json     # allensdk EcephysProjectCache
    ecephys_cache/session_<id>/session_<id>.nwb
  dandi/
    000409/                         # optional NWB mirror of the BWM
    000021/                         # Allen Visual Coding NWB via dandi
```

## 1. IBL Brain-wide Map via ONE (recommended)

```bash
pip install ONE-api ibllib iblatlas
pip install git+https://github.com/int-brain-lab/paper-brain-wide-map.git   # provides `brainwidemap`
python - <<'EOF'
from one.api import ONE
from brainwidemap import bwm_query
one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international",
          silent=True, cache_dir="data/ibl/one_cache")
df = bwm_query(one)                       # one row per insertion: eid, pid, subject, lab, probe_name, date
df.to_csv("data/ibl/bwm_sessions.csv", index=False)
print(df.lab.value_counts())
EOF
```

Per insertion (see `ibl_bwm.loaders.load_ibl_session`):

```python
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)   # adds acronym, x/y/z, label (QC)
trials = one.load_object(eid, "trials")                       # choice, contrastLeft/Right, probabilityLeft,
                                                              # feedbackType, stimOn_times, firstMovement_times, feedback_times
```

Or with the downloader: `python scripts/download_data.py ibl --sample` (3 insertions) /
`python scripts/download_data.py ibl --all` (everything; ~3 TB, days).

## 2. IBL Brain-wide Map via DANDI (NWB)

```bash
pip install dandi
dandi download --output-dir data/dandi "https://dandiarchive.org/dandiset/000409/draft" --existing skip
# or a single asset path via `dandi ls` first; the downloader's `dandi --dandiset 000409 --sample` pulls
# the first asset only.
```

## 3. IBL Brain-wide Map via AWS

```bash
aws s3 ls --no-sign-request s3://ibl-brain-wide-map-public/
aws s3 sync --no-sign-request s3://ibl-brain-wide-map-public/aggregates/ data/ibl/aggregates/
```

## 4. Allen Visual Coding – Neuropixels

```bash
pip install allensdk
python - <<'EOF'
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest="data/allen/ecephys_cache/manifest.json")
sessions = cache.get_session_table()
sessions.to_csv("data/allen/sessions.csv")
sid = sessions.index[0]
session = cache.get_session_data(sid)          # downloads the NWB (~2 GB)
print(session.units[["ecephys_structure_acronym"]].value_counts().head())
EOF
```
Alternative: `dandi download https://dandiarchive.org/dandiset/000021` or
`aws s3 sync --no-sign-request s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/ data/allen/ecephys_cache/`.

## 5. Atlas

`pip install iblatlas` — `from iblatlas.regions import BrainRegions; br = BrainRegions();
br.acronym2acronym(acronyms, mapping="Beryl")` gives the Beryl parcellation used in the BWM.
