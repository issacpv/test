# Data acquisition — radiology-report-weak-supervision-audit

Everything under `data/` is git-ignored. Most sources are PhysioNet credentialed: complete CITI "Data or Specimens Only Research" training, get PhysioNet credentialing, and sign each project's DUA. Store data only on systems approved under your DUA. Never send reports or images to external services.

## 1. PhysioNet resources (credentialed)

Set credentials in the environment (never in files):

```bash
export PHYSIONET_USER=... ; export PHYSIONET_PASS=...
python scripts/download_data.py --source physionet --project mimic-cxr-jpg --version 2.1.0 --sample   # label/metadata tables only
python scripts/download_data.py --source physionet --project chest-imagenome --version 1.0.0 --include gold_dataset
python scripts/download_data.py --source physionet --project reflacx-xray-localization --version 1.0.0
python scripts/download_data.py --source physionet --project ms-cxr --version 1.1.0
python scripts/download_data.py --source physionet --project vindr-cxr --version 1.0.0 --include annotations
python scripts/download_data.py --source physionet --project mimiciv --version 3.1 --include 'hosp/(patients|admissions|transfers)' 'icu/icustays'
```

Equivalent `wget` (PhysioNet's documented route):

```bash
wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

Images: only the expert-labelled studies are needed for RQ5; build the study list first (`outputs/expert_master.parquet`) and pass it with `--study-list` to fetch just those JPGs.

Expected layout:

```
data/physionet/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz
data/physionet/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-negbio.csv.gz
data/physionet/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
data/physionet/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz
data/physionet/mimic-cxr-jpg/2.1.0/files/p10/p10000032/s50414267/*.jpg
data/physionet/mimic-cxr/2.1.0/files/p10/p10000032/s50414267.txt           # reports (project mimic-cxr)
data/physionet/chest-imagenome/1.0.0/gold_dataset/*.txt|*.csv
data/physionet/reflacx-xray-localization/1.0.0/main_data/*/metadata_phase_*.csv
data/physionet/ms-cxr/1.1.0/MS_CXR_Local_Alignment_v1.1.0.csv
data/physionet/vindr-cxr/1.0.0/annotations/*.csv
data/physionet/mimiciv/3.1/hosp/{patients,admissions,transfers}.csv.gz ; icu/icustays.csv.gz ; (mimic-iv-ed for edstays)
```

## 2. CheXpert (free registration, Stanford AIMI)

Register at https://stanfordaimi.azurewebsites.net/ (CheXpert / CheXpert Plus), download the validation set (200 studies, 3 radiologists) and the test set labels (500 studies) plus `train.csv`/`valid.csv`. Place under `data/chexpert/`.

## 3. PadChest (free registration, BIMCV)

Request access at https://bimcv.cipf.es/bimcv-projects/padchest/ and download `PADCHEST_chest_x_ray_images_labels_160K.csv` (the `MethodLabel` column marks physician-labelled vs NLP-labelled reports). Images are optional. Place under `data/padchest/`.

## 4. NIH ChestX-ray14 and adjudicated labels (open)

Images and `Data_Entry_2017.csv` from https://nihcc.app.box.com/v/ChestXray-NIHCC. The radiologist-adjudicated labels (Majkowska et al., 2020, Radiology) are distributed with that paper's supplementary resources / the NIH Box folder ("adjudicated labels"); place them under `data/nih/`. `python scripts/download_data.py --source nih` prints the current instructions.

## Derived products (aggregate only; never contain report text)

```
outputs/
  labels/mimic_rule_labels.parquet, mimic_chexbert.parquet, mimic_llm_local.parquet
  features/report_features.parquet
  expert_master.parquet                       # harmonised expert image labels with study keys
  agreement/noise_matrices.csv, kappa.csv
  differential/odds_ratios.csv
  simulation/fairness_gaps.csv
  benchmark/model_rankings.csv
```
