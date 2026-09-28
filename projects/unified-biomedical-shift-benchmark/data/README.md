# Data acquisition and cache export

Nothing in this folder is committed except this file. BioShift never reads raw signals: every domain is consumed through a small **cache** (`data/cache/<task>/<domain>.npz` + `_meta.csv`) that the sibling projects export, and a **manifest** of group-level splits (`manifests/`). The raw downloads below are needed only to build those caches.

## Expected layout

```
data/
  ptbxl/ chapman/ ningbo/ georgia/ cpsc/ code15/        # ECG (open)
  chbmit/ siena/ helsinki/ tusz/                         # EEG (open / registration)
  mimic_iv/ eicu/ hirid/ aumcdb/                         # ICU (credentialed / DUA)
  echonet_dynamic/ echonet_pediatric/ camus/             # echo (registration)
  cache/
    icu_mortality_48h/mimic_iv.npz  mimic_iv_meta.csv    # X (n, d) float32, y (n,), groups (n,)
    ecg_dx12/ptbxl.npz ...                               # y is (n, 12) int for multilabel
    eeg_seizure_event/chbmit.npz ...                     # + record_id, t_start arrays
    echo_ef/echonet_dynamic.npz ...                      # y = EF (%), optional pred_std later
manifests/<task>/<domain>_splits.csv                     # group,split  (committed)
```

## Downloads

| Domain | Command | Access |
|---|---|---|
| PTB-XL, Chapman/Ningbo, Georgia, CPSC | `python scripts/download_data.py --domain ptbxl [--sample]` (wget mirror of `https://physionet.org/files/ptb-xl/1.0.3/`; likewise `ecg-arrhythmia/1.0.0/`, `challenge-2021/1.0.3/`) | Open, no login |
| CODE-15% | `python scripts/download_data.py --domain code15 [--sample]` (Zenodo API record 4916206; 18 HDF5 shards + `exams.csv`) | Open (CC-BY) |
| CHB-MIT, Siena | `--domain chbmit`, `--domain siena` (PhysioNet files) | Open |
| Helsinki neonatal | `--domain helsinki` (Zenodo search by title; Stevenson et al., 2019) | Open (CC-BY) |
| TUSZ v2.0.3 | `--domain tusz` prints the rsync command; needs the TUH data-use form | Free registration |
| MIMIC-IV v3.1, eICU-CRD v2.0, HiRID v1.1.1 | `PHYSIONET_USER=... PHYSIONET_PASS=... python scripts/download_data.py --domain mimic_iv` | PhysioNet credentialed (CITI + DUA) |
| AmsterdamUMCdb v1.0.2 | `--domain aumcdb` prints instructions | End-user licence |
| EchoNet-Dynamic / EchoNet-Pediatric | `--domain echonet_dynamic` prints instructions (Stanford AIMI research-use agreement) | Free registration + agreement |
| CAMUS | `--domain camus` prints instructions | Free registration |

## Cache export (per modality)

Each sibling project owns the modality-specific preprocessing; BioShift only fixes the *contract*:

- **ICU** (`icu-model-transportability`): 24-h window features on the shared 38-concept ontology (mean/min/max/last/count + missingness) -> `X`; labels per task -> `y`; `groups = subject_id` (stay-level rows, patient-level groups); `meta`: `sex`, `age_band` (10-year bands to match AUMCdb).
- **ECG** (`ecg-cross-dataset-generalization`): either handcrafted features or the penultimate embedding of the source-trained 1D-ResNet (one cache per source model; name them `<domain>__emb_<model>.npz` and point the adapter at them) -> `X`; 12 harmonised classes under the *lenient* mapping -> `y` (n, 12); `groups = patient_id`; `meta`: `sex`, `age_band` (<40, 40-64, >=65).
- **EEG** (`cross-dataset-seizure-generalization`): 4-s windows, 2-s step, 18-pair bipolar 256 Hz; spectral + Hjorth features (or FM embeddings) -> `X`; window label -> `y`; `groups = subject_id`; `meta`: `record_id`, `t_start` (s), `sex`, `age_band` (neonatal / pediatric / adult).
- **Echo** (`echonet-ef-uncertainty`): per-video embedding (or the sibling's uncertainty model's features) -> `X`; EF (%) -> `y`; `groups = patient_id` (EchoNet exposes one video per study; CAMUS `patientXXXX`); `meta`: `sex`, `age_band` (pediatric / adult).

Export skeleton (numpy only):

```python
np.savez("data/cache/icu_mortality_48h/mimic_iv.npz", X=X.astype("float32"), y=y.astype("int8"), groups=subject_id.astype(str))
meta.to_csv("data/cache/icu_mortality_48h/mimic_iv_meta.csv", index=False)
```

Then `python scripts/download_data.py --make-manifests` writes the group-level split files; overwrite them with official splits where they exist (PTB-XL `strat_fold`, TUSZ train/dev/eval, EchoNet `FileList.csv` split).

## Sizes

ECG open sets ~10 GB total (CODE-15% ~7 GB); CHB-MIT ~42 GB; Siena ~20 GB; Helsinki ~4 GB; TUSZ ~70 GB; MIMIC-IV ~30 GB, eICU ~20 GB, HiRID ~40 GB, AUMCdb ~80 GB; EchoNet-Dynamic ~7 GB, EchoNet-Pediatric ~4 GB, CAMUS ~3 GB. Caches are small (tens of MB per domain).

## Never commit

Raw data, caches, embeddings or any patient-level table. Only `manifests/` (identifiers already public in the source datasets) and aggregate results are committed. Credentialed data (MIMIC, eICU, HiRID) must not be sent to third-party APIs.
