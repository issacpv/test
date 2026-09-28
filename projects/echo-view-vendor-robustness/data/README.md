# Data acquisition — Echo-View-Shift

Nothing here is committed. Sources have mixed access; MIMIC-IV-ECHO is credentialed.

## 1. EchoNet-Dynamic (free registration + DUA)

Register at https://echonet.github.io/dynamic/ and download `EchoNet-Dynamic.zip` (A4C AVI videos + `FileList.csv`, `VolumeTracings.csv`). Used as a single-view (A4C) source and for the RQ5 EF-impact analysis.

## 2. CAMUS (registration)

Download from https://www.creatis.insa-lyon.fr/Challenge/camus/ . Provides A4C and A2C frames (ED/ES) with view labels; GE vendor, single center.

## 3. TMED-2 (free registration + DUA)

Request access at https://tmed.cs.tufts.edu/tmed_v2.html . Multi-view labelled 2D images (PLAX/PSAX/A2C/A4C) + aortic-stenosis severity.

## 4. TTE47 (open benchmark)

Available at https://thrive-centre.com/datasets/TTE47 (47 fine-grained views). Used for the fine-grained crosswalk and as an additional target.

## 5. MIMIC-IV-ECHO (credentialed)

Complete PhysioNet CITI training and sign the DUA for https://physionet.org/content/mimic-iv-echo/0.1/ . Export credentials (never commit):

```bash
export PHYSIONET_USERNAME="your_user"
export PHYSIONET_PASSWORD="your_pass"
python scripts/download_data.py --dataset mimic-iv-echo --out data/mimic-iv-echo --sample
```
DICOM studies without curated view labels; a stratified subset must be human-view-labeled inside the credentialed environment. The DICOM `Manufacturer` tag gives ground-truth vendor.

## Expected layout

```
data/
  echonet-dynamic/
    FileList.csv  VolumeTracings.csv  Videos/*.avi
  camus/
    patientXXXX/*.mhd|*.raw|*.png   # per-patient A4C/A2C ED/ES
  tmed2/
    images/*.png  labels.csv
  tte47/
    images/...  labels.csv
  mimic-iv-echo/
    files/...     # DICOM (credentialed, gitignored)
    view_labels_sample.csv   # your human-labeled stratified subset
frames/           # extracted, harmonised frames (gitignored)
```
