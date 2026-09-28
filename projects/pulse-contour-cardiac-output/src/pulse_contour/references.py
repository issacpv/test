"""Reference-standard stroke volume / cardiac output from three independent sources.

1. **Echo LVOT method** (MIMIC-III echo reports; MIMIC-IV-Echo measurements):
   SV = pi * (LVOT diameter / 2)^2 * LVOT VTI. ``parse_echo_measurements`` reads the
   semi-structured measurement lines of a report; field names vary between eras, so
   the regexes are deliberately tolerant and every parsed report keeps the raw units.
2. **Thermodilution / continuous CO from PA catheters** (MIMIC chartevents):
   ``cardiac_output_items`` finds the relevant ``d_items`` rows by label so nothing is hard-coded.
3. **Oesophageal Doppler and FloTrac/EV1000/Vigileo** (VitalDB): ``vitaldb_tracks`` pulls the
   numeric tracks through the public VitalDB REST API (or the ``vitaldb`` package if installed).

``align_reference`` matches windowed pulse-contour estimates to reference time points.
"""
from __future__ import annotations

import io
import re
from typing import Iterable

import numpy as np
import pandas as pd

VITALDB_API = "https://api.vitaldb.net"
VITALDB_TRACKS = {
    "abp": "SNUADC/ART",           # 500 Hz arterial waveform
    "ev1000_sv": "EV1000/SV", "ev1000_co": "EV1000/CO", "ev1000_svv": "EV1000/SVV",
    "vigileo_sv": "Vigileo/SV", "vigileo_co": "Vigileo/CO",
    "cardioq_sv": "CardioQ/SV", "cardioq_co": "CardioQ/CO", "cardioq_ftc": "CardioQ/FTc",
    "nibp_sbp": "Solar8000/NIBP_SBP", "nibp_dbp": "Solar8000/NIBP_DBP",
}

_NUM = r"([\d]+(?:\.\d+)?)"
ECHO_FIELDS: dict[str, re.Pattern] = {
    "lvot_diam_cm": re.compile(rf"LVOT\s*(?:diam(?:eter)?|d)\s*[:=]?\s*{_NUM}\s*(cm|mm)", re.IGNORECASE),
    "lvot_vti_cm": re.compile(rf"LVOT\s*VTI\s*[:=]?\s*{_NUM}\s*(cm)?", re.IGNORECASE),
    "lvef_pct": re.compile(rf"(?:Ejection\s*Fraction|LVEF|EF)\s*[:=]?\s*[><=]*\s*{_NUM}\s*%", re.IGNORECASE),
    "stroke_volume_ml": re.compile(rf"(?:Stroke\s*Volume|SV)\s*[:=]?\s*{_NUM}\s*(?:ml|mL)", re.IGNORECASE),
    "cardiac_output_lmin": re.compile(rf"(?:Cardiac\s*Output|CO)\s*[:=]?\s*{_NUM}\s*(?:L/min|l/min)", re.IGNORECASE),
}


def parse_echo_measurements(text: str) -> dict[str, float]:
    """Extract numeric echo measurements (cm, %, mL, L/min) from a report; mm are converted to cm."""
    out: dict[str, float] = {}
    for key, rx in ECHO_FIELDS.items():
        m = rx.search(text)
        if not m:
            continue
        val = float(m.group(1))
        if key == "lvot_diam_cm" and m.group(2).lower() == "mm":
            val /= 10.0
        out[key] = val
    return out


def stroke_volume_from_lvot(diam_cm: float, vti_cm: float) -> float:
    """SV (mL) = pi * (d/2)^2 * VTI, with d and VTI in cm (1 cm^3 = 1 mL)."""
    return float(np.pi * (diam_cm / 2.0) ** 2 * vti_cm)


def echo_reference(measurements: dict[str, float]) -> float:
    """Prefer the LVOT-derived SV; fall back to a reported SV; NaN otherwise."""
    if "lvot_diam_cm" in measurements and "lvot_vti_cm" in measurements:
        return stroke_volume_from_lvot(measurements["lvot_diam_cm"], measurements["lvot_vti_cm"])
    return float(measurements.get("stroke_volume_ml", np.nan))


def cardiac_output_items(d_items: pd.DataFrame, pattern: str = r"cardiac output|stroke volume|\bC\.?O\b") -> pd.DataFrame:
    """Rows of MIMIC ``d_items`` whose label matches ``pattern`` (thermodilution, CCO, Fick, ...) for manual verification."""
    m = d_items["label"].astype(str).str.contains(pattern, case=False, regex=True)
    cols = [c for c in ("itemid", "label", "category", "unitname", "linksto") if c in d_items]
    return d_items.loc[m, cols].reset_index(drop=True)


def vitaldb_tracks(caseid: int, tracks: Iterable[str] = ("CardioQ/SV", "EV1000/SV", "SNUADC/ART"),
                   api: str = VITALDB_API, trk_index: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Download numeric/waveform tracks for one VitalDB case as ``{track: DataFrame(Time, value)}``.

    Uses the ``vitaldb`` package when installed; otherwise the REST API (``/trks`` lists track ids,
    ``/{tid}`` returns CSV). Network access is required; the function is not used by the tests.
    """
    try:
        import vitaldb  # type: ignore

        vf = vitaldb.VitalFile(caseid, list(tracks))
        return {tr: pd.DataFrame(vf.to_numpy([tr], 0), columns=[tr]).rename_axis("Time").reset_index() for tr in tracks}
    except ImportError:
        pass
    import requests

    if trk_index is None:
        trk_index = pd.read_csv(f"{api}/trks")
    out: dict[str, pd.DataFrame] = {}
    for tr in tracks:
        row = trk_index[(trk_index["caseid"] == caseid) & (trk_index["tname"] == tr)]
        if row.empty:
            continue
        r = requests.get(f"{api}/{row['tid'].iloc[0]}", timeout=120)
        r.raise_for_status()
        out[tr] = pd.read_csv(io.StringIO(r.text))
    return out


def align_reference(est: pd.DataFrame, ref: pd.DataFrame, window_s: float = 60.0,
                    t_est: str = "t_center", v_est: str = "sv", t_ref: str = "t", v_ref: str = "sv_ref") -> pd.DataFrame:
    """For every reference measurement, the median estimate within +-window_s; rows without estimates are dropped."""
    rows = []
    te, ve = est[t_est].to_numpy(float), est[v_est].to_numpy(float)
    for t, v in zip(ref[t_ref].to_numpy(float), ref[v_ref].to_numpy(float)):
        m = (np.abs(te - t) <= window_s) & np.isfinite(ve)
        if m.any():
            rows.append({"t": t, "sv_ref": v, "sv_est": float(np.median(ve[m])), "n_windows": int(m.sum())})
    return pd.DataFrame(rows, columns=["t", "sv_ref", "sv_est", "n_windows"])
