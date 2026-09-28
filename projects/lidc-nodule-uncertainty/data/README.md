# Data acquisition

Nothing in this directory is committed. Record the **download date** and any dataset
version for every source; TCIA collections and Zenodo records are both versioned.

Check free disk before starting: LIDC-IDRI alone is ~125 GB.

## Expected directory layout

```
data/
├── lidc/
│   ├── series_manifest.json          # written by scripts/download_data.py
│   ├── zips/<SeriesInstanceUID>.zip  # only when using the REST route
│   └── LIDC-IDRI/                    # extracted DICOM tree (NBIA Data Retriever)
│       └── LIDC-IDRI-0001/
│           └── <study>/<series>/*.dcm + *.xml     # XML holds the 4 readers
├── luna16/
│   ├── annotations.csv
│   ├── candidates_V2.csv
│   └── subset0/ ... subset9/         # .mhd + .raw
├── dlcs/                             # Duke Lung Cancer Screening
│   ├── <annotation csv / json>
│   └── <image volumes>
├── nlst/                             # only if a CDAS request is approved
└── derivatives/
    ├── nodule_table.csv              # one row per nodule: soft labels + disagreement
    ├── features_simple.csv
    ├── features_pyradiomics.csv
    └── models/
```

## 1. LIDC-IDRI (open, no registration) — the primary dataset

The radiologist annotations are the point of this project: up to four readers per
nodule, malignancy 1–5 plus eight further semantic ratings, shipped as XML alongside
the DICOM.

**Route A — NBIA Data Retriever (recommended for the full collection).** More
reliable than scripted REST calls for 125 GB:

1. Open https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Download the collection's `.tcia` manifest.
3. Install the NBIA Data Retriever, open the manifest, and download into
   `data/lidc/LIDC-IDRI/`.

**Route B — REST API (good for a small sample).** What the script does:

```bash
# Metadata only: writes data/lidc/series_manifest.json
python scripts/download_data.py --lidc-manifest

# A 5-series sample, images included
python scripts/download_data.py --sample
```

The underlying calls:

```
GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=LIDC-IDRI&format=JSON
GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<uid>
    -> a zip of DICOM files
```

The script **refuses** `--images` without `--limit`, so a stray command cannot start
a 125 GB download.

### Configure pylidc

`pylidc` needs an absolute path in `~/.pylidcrc`; a relative path makes it find zero
scans with no error message.

```bash
python scripts/download_data.py --configure-pylidc /abs/path/to/data/lidc/LIDC-IDRI
```

which writes:

```ini
[dicom]
path = /abs/path/to/data/lidc/LIDC-IDRI
warn = True
```

Verify, then build the nodule table:

```bash
python -c "import pylidc as pl; print(pl.query(pl.Scan).count())"   # expect ~1018
```

```python
from lidc_uq.annotations import load_pylidc_nodules, build_nodule_table

nodules = load_pylidc_nodules(limit=None, min_raters=3, load_masks=False)
table = build_nodule_table(nodules)
table.to_csv("data/derivatives/nodule_table.csv", index=False)
```

Run with `limit=5` first. Iterating every scan with `load_masks=True` is slow and
memory-hungry.

**Record the exclusions.** `min_raters=3` follows LUNA16, but nodules seen by fewer
readers are the most uncertain of all, and dropping them silently biases a
disagreement analysis toward agreement. Log the count at every `min_raters` value
from 1 to 4 before choosing one.

**Alternative annotation source.** TCIA also hosts a DICOM-SR re-encoding of the
LIDC annotations ("Standardized representation of the LIDC annotations using
DICOM"), useful if you prefer DICOM-native tooling to `pylidc`.

## 2. LUNA16 (open) — detection arm and standard folds

```bash
pip install zenodo-get
zenodo_get 3723295 -o data/luna16
```

Or fetch individual files from https://zenodo.org/records/3723295 :
`annotations.csv`, `candidates_V2.csv`, `subset0.zip` … `subset9.zip`.

Images are `.mhd`/`.raw`, readable with SimpleITK:

```python
import SimpleITK as sitk
img = sitk.ReadImage("data/luna16/subset0/<uid>.mhd")
array = sitk.GetArrayFromImage(img)          # (z, y, x)
spacing = tuple(reversed(img.GetSpacing()))  # -> (z, y, x) to match the array
```

**Keep LUNA16's official 10-fold split** for any detection number you report, or the
result is not comparable with the literature. Note that LUNA16 excludes scans with
slice thickness > 2.5 mm and nodules accepted by fewer than 3 of 4 readers, so it is
*already* filtered toward agreement — which is exactly why the primary disagreement
analysis runs on full LIDC, not on LUNA16.

## 3. Duke Lung Cancer Screening (DLCS) — external transfer cohort

```bash
pip install zenodo-get
zenodo_get 10.5281/zenodo.10782891 -o data/dlcs
```

Dataset paper: "The Duke Lung Cancer Screening (DLCS) Dataset: A Reference Dataset
of Annotated Low-dose Screening Thoracic CT", *Radiology: Artificial Intelligence*,
2025, doi:10.1148/ryai.240248

What it is: 2061 screening LDCT scans with ~3187 semi-automatically annotated
nodules; the released subset is 1613 volumes with 2487 nodules, annotated as **3D
bounding boxes**, with diagnostic labels.

**Two things to record per nodule, both of which affect the transfer analysis:**

1. **Annotation group (1–4).** Groups 1 and 2 were annotated entirely by the
   automated algorithm; group 3 was algorithm-generated and manually confirmed;
   group 4 was annotated manually. Radiologist spot-checking put the semi-automatic
   accuracy above 90%. Stratify every DLCS result by group, and restrict the primary
   transfer analysis to groups 3–4.
2. **That there are no multi-reader malignancy ratings.** DLCS cannot supply a
   disagreement *target*. It is the cohort where a LIDC-trained model is *applied*,
   with a proxy target, and the write-up must say so.

Bounding boxes, not masks, means shape and boundary features must be recomputed
inside a box-derived ROI rather than a contour. Document the ROI convention, and
run the LIDC model on box-derived ROIs too when comparing — otherwise the transfer
drop measures the ROI change rather than the cohort change.

## 4. NLST (application required) — optional

NCI Cancer Data Access System: https://cdas.cancer.gov/nlst/

- Submit a data request describing the project. Approval typically takes weeks.
- Imaging and the clinical/outcome tables are **separate** requests.
- Reference: Aberle et al., 2011, *New England Journal of Medicine*, "Reduced
  lung-cancer mortality with low-dose computed tomographic screening".

Not scriptable; CDAS provides its own download tool after approval. Respect the
agreement's redistribution terms — do not include NLST individual-level data in any
released derivative.

## Credentialed sources

**This project uses none.** If the wider programme later adds PhysioNet resources,
read credentials from environment variables only:

```bash
export PHYSIONET_USER=...      # example pattern; not used by this project
export PHYSIONET_PASS=...
```

and remember that credentialed PhysioNet data may not be sent to third-party LLM
APIs except as permitted by PhysioNet's responsible-use policy.

## Provenance checklist before analysis

- [ ] LIDC download date and TCIA collection version recorded
- [ ] `pylidc` scan count verified (~1018) and `~/.pylidcrc` path absolute
- [ ] Reader-count distribution tabulated; exclusion counts at `min_raters` 1–4 logged
- [ ] Slice-thickness and manufacturer distributions tabulated — these are the
      acquisition confounders
- [ ] LUNA16 official fold assignment preserved alongside any derived table
- [ ] DLCS annotation group recorded per nodule; ROI convention documented
- [ ] Zenodo record versions recorded for LUNA16 and DLCS
- [ ] Feature backend (`simple` vs `pyradiomics`) recorded per feature table, and
      the two never mixed within one model
