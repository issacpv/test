# Data acquisition — CGM-Regimen-Shift

Nothing here is committed. Access varies by dataset; none is committed to git.

## 1. OhioT1DM (DUA)

Request access via the form at http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html . Ships XML per subject (2018 and 2020 cohorts) with CGM (5 min), basal/bolus insulin, carbs, and some sensor/activity fields. Open-loop regimen. Preserve the official train/test split.

## 2. AZT1D (open, Mendeley Data)

```bash
python scripts/download_data.py --dataset azt1d --out data/azt1d
```
25 patients on Tandem Control-IQ AID; CGM + pump + fine-grained bolus + device mode (regular/sleep/exercise). Openly licensed. (If the automated fetch fails, download the archive manually from https://data.mendeley.com/datasets/gk9m674wcx/1 into `data/azt1d/`.)

## 3. DiaTrend (registration, Synapse)

Register on Synapse and follow the data-availability link in Prioleau et al. 2023 (doi:10.1038/s41597-023-02469-5). 54 patients; CGM + pump; mixed regimen.

## 4. OpenAPS Data Commons (data request)

Submit a data request via https://openaps.org/outcomes/ (OpenAPS Data Commons). DIY closed-loop data (OpenAPS/AndroidAPS/Loop); CGM + insulin + algorithm decisions; heterogeneous - see the audit module.

## 5. T1DEXI (optional, JAEB/Vivli)

Register at https://public.jaeb.org/ for T1DEXI (CGM + pump + wearable HR during exercise).

## Expected layout

```
data/
  ohiot1dm/
    2018/train/*.xml  2018/test/*.xml  2020/train/*.xml  2020/test/*.xml
  azt1d/
    *.csv                       # per-subject CGM + pump + device mode
  diatrend/
    Subject*.xlsx|*.csv
  openaps/
    <user>/entries.json  <user>/treatments.json   # Nightscout-style exports
  t1dexi/
    ...
```

Regimen labels are assigned by `src/cgm_shift/regimen.py` from device/algorithm metadata.
