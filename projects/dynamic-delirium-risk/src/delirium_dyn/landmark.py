"""Landmark datasets with alternative label schemes and delirium/coma-free days.

A landmark row is (stay, window t) with features computed from data before ``w_start`` of window t and a label
taken from the state at window ``t + horizon``.  Three schemes are derived from the same rows so that the
effect of label handling (H1 in the README) is a paired comparison:

* ``multistate``            label in {normal, delirium, coma, unscreened, discharged, dead};
* ``binary_drop_coma``      label = 1 if delirium, 0 if normal; rows with coma/unscreened future removed
                            (terminal futures kept as 0 unless ``drop_terminal``);
* ``binary_coma_negative``  label = 1 if delirium else 0 (coma/unscreened/terminal all 0).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TERMINAL = ("discharged", "dead")
SCHEMES = ("multistate", "binary_drop_coma", "binary_coma_negative")


def build_landmark_dataset(
    windows: pd.DataFrame,
    horizon: int = 1,
    first_landmark: int = 2,
    scheme: str = "multistate",
    drop_terminal: bool = False,
) -> pd.DataFrame:
    """Landmark rows from a window-state table (``stay_id, window_idx, state`` at least).

    Returns the landmark row (current window's columns prefixed ``cur_``) plus ``future_state`` and ``label``.
    Landmarks whose current state is terminal are excluded; landmarks without a window at ``t + horizon`` are
    excluded (this cannot happen when the terminal row is present, since it is the last window).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    w = windows.sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
    state_of = {(s, i): st for s, i, st in zip(w["stay_id"], w["window_idx"], w["state"])}
    rows = []
    for _, r in w.iterrows():
        t = int(r["window_idx"])
        if t < first_landmark or r["state"] in TERMINAL:
            continue
        # future: the state `horizon` windows ahead, or the terminal state if the stay ended before
        fut = None
        for h in range(horizon, 0, -1):
            fut = state_of.get((r["stay_id"], t + h))
            if fut is not None:
                break
        if fut is None:
            continue
        rows.append({**{f"cur_{k}": v for k, v in r.items()}, "stay_id": r["stay_id"], "landmark_idx": t, "future_state": fut})
    lm = pd.DataFrame(rows)
    if lm.empty:
        return lm
    if scheme == "multistate":
        lm["label"] = lm["future_state"]
    elif scheme == "binary_drop_coma":
        keep = ~lm["future_state"].isin(["coma", "unscreened"])
        if drop_terminal:
            keep &= ~lm["future_state"].isin(TERMINAL)
        lm = lm[keep].copy()
        lm["label"] = (lm["future_state"] == "delirium").astype(int)
    else:
        lm["label"] = (lm["future_state"] == "delirium").astype(int)
    return lm.reset_index(drop=True)


def label_scheme_summary(windows: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Row counts and delirium prevalence under each scheme (paired on the same landmarks)."""
    out = []
    for s in SCHEMES:
        lm = build_landmark_dataset(windows, horizon=horizon, scheme=s)
        if s == "multistate":
            prev = float((lm["label"] == "delirium").mean()) if len(lm) else np.nan
        else:
            prev = float(lm["label"].mean()) if len(lm) else np.nan
        out.append({"scheme": s, "n_rows": int(len(lm)), "delirium_prevalence": prev})
    return pd.DataFrame(out)


def delirium_coma_free_days(windows: pd.DataFrame, window_hours: int = 12, horizon_days: int = 14) -> pd.DataFrame:
    """Days alive without delirium or coma within ``horizon_days`` of ICU admission, per stay.

    Windows in ``normal`` or ``unscreened`` count as free; ``delirium``/``coma`` do not; after discharge the
    remaining days count as free; death sets the value to 0 (convention used in sedation trials).
    """
    per_day = 24 / window_hours
    n_max = int(horizon_days * per_day)
    out = []
    for sid, g in windows.sort_values("window_idx").groupby("stay_id"):
        g = g[g["window_idx"] < n_max]
        died = (g["state"] == "dead").any()
        if died:
            out.append({"stay_id": sid, "dcfd": 0.0, "died": True})
            continue
        non_term = g[~g["state"].isin(TERMINAL)]
        free = non_term["state"].isin(["normal", "unscreened"]).sum()
        after = n_max - len(non_term)  # windows after discharge within the horizon
        out.append({"stay_id": sid, "dcfd": float((free + max(after, 0)) / per_day), "died": False})
    return pd.DataFrame(out)


def transition_counts(windows: pd.DataFrame) -> pd.DataFrame:
    """Matrix of observed window-to-window transitions (rows: from, columns: to), per stay sequence."""
    w = windows.sort_values(["stay_id", "window_idx"])
    pairs = []
    for _, g in w.groupby("stay_id"):
        st = g["state"].tolist()
        pairs.extend(zip(st[:-1], st[1:]))
    if not pairs:
        return pd.DataFrame()
    df = pd.DataFrame(pairs, columns=["from", "to"])
    return pd.crosstab(df["from"], df["to"])
