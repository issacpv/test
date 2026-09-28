# Data acquisition

Nothing in this folder is committed except this file. All datasets are open (OpenNeuro, CC0 / CC-BY-style licences) and in BIDS-EEG (Pernet et al., 2019, *Sci Data*), so a single loader (`mne-bids`) covers all of them.

## Expected layout

```
data/
  candidates.csv                 # produced by --scan: OpenNeuro EEG datasets mentioning neurofeedback (screen manually)
  ds002336/                      # XP1 (Lioi et al., 2020): sub-xp1XX/ses-*/eeg/*.vhdr|.set, beh/, derivatives (online NF scores)
  ds002338/                      # XP2 (Lioi et al., 2020): 20 subjects, bimodal NF, 1-D vs 2-D feedback
  ds005846/  ds005878/           # parietal-alpha down-regulation NF in immersive VR (Front. Neurosci., 2025)
  derived/
    manifests/sessions.csv       # dataset, subject, session, run, task, fs, n_channels, target channels, baseline window, block edges
    feedback/<dataset>/<sub>_<ses>_<spec_id>.parquet   # recomputed feedback per specification (safe to regenerate)
    online/<dataset>/<sub>_<ses>.parquet               # the online feedback values as shipped by the dataset (when available)
```

## 1. Discover candidates (OpenNeuro GraphQL API)

```bash
python scripts/download_data.py --scan
```

This pages through every EEG dataset on OpenNeuro (`datasets(modality: "EEG")`, cursor pagination) and keeps those whose name or README mentions neurofeedback / closed-loop / BCI. Screen `data/candidates.csv` by hand for the inclusion criteria: (i) EEG-based feedback (not fMRI-only), (ii) raw continuous EEG with the training blocks marked (events.tsv or the paper's timing), (iii) ideally the online feedback values or the online pipeline description.

## 2. Download (public S3 bucket over HTTPS, no credentials)

```bash
python scripts/download_data.py --dataset ds002338 --sample      # metadata + first subject (~hundreds of MB)
python scripts/download_data.py --dataset ds002336 ds002338 ds005846 ds005878
```

Equivalent tools: `pip install openneuro-py && openneuro-py download --dataset ds002338`, or `aws s3 sync --no-sign-request s3://openneuro.org/ds002338 data/ds002338`, or `datalad install https://github.com/OpenNeuroDatasets/ds002338.git`.

Sizes: XP1/XP2 include simultaneous fMRI and anatomical MRI; EEG-only needs are a few GB per dataset (`--sample` first, then filter to `*/eeg/*` if disk is tight: `aws s3 sync --exclude "*" --include "*/eeg/*" ...`). The 2025 VR datasets are EEG-only.

## 3. What each dataset provides for the multiverse

| Dataset | Feedback target (online) | Sessions/blocks | Online values shared? | Notes |
|---|---|---|---|---|
| ds002336 (XP1) | Motor-imagery sensorimotor (mu/beta) desynchronisation over C3 (EEG-NF), fMRI-NF, bimodal | 1 session, several NF runs per condition | NF scores documented in the data descriptor; check `derivatives/` | 64-ch EEG inside the MR scanner: gradient/BCG artifact removal is a *pre-multiverse* step (fixed pipeline) |
| ds002338 (XP2) | Bimodal EEG-fMRI motor-imagery NF; 1-D vs 2-D visual feedback | 1 session, 3 NF runs | as above | 20 subjects: the largest n here |
| ds005846 / ds005878 | Parietal alpha *down*-regulation in VR | multiple blocks per session (see events.tsv) | check `beh/` and README | direction = down; reward rule inverted |

Fill `derived/manifests/sessions.csv` from `participants.tsv`, `*_events.tsv` and `*_eeg.json` (sampling rate, reference, channel names) before running the multiverse; the code needs only `fs`, channel names, block edges and a baseline window.

## 4. Loading

```python
import mne_bids, mne
bp = mne_bids.BIDSPath(subject="xp201", task="MIpre", root="data/ds002338", datatype="eeg")
raw = mne_bids.read_raw_bids(bp, verbose=False).load_data()
raw.resample(250)                      # fixed pre-step; the multiverse starts at nf_multiverse.compute_feedback
eeg, ch_names, fs = raw.get_data() * 1e6, raw.ch_names, raw.info["sfreq"]
```

Keep MR-artifact correction (XP1/XP2) and resampling *outside* the specification space; they are documented once and applied identically to every specification.

## Licences / ethics

OpenNeuro datasets are de-identified and openly licensed; cite the data descriptors (Lioi et al., 2020, *Sci Data* for XP1/XP2; the 2025 *Frontiers in Neuroscience* paper for ds005846/ds005878) and OpenNeuro. Do not commit data or derived per-participant time series; `derived/` is git-ignored except manifests.
