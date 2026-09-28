"""Session tables, scanner-transition detection and running variables.

OASIS-3 identifiers encode the subject and the number of days since study entry
(``OAS30001_MR_d0129``); calendar dates are not released. The design used for OASIS-3 is
therefore a *subject-anchored event-time* regression discontinuity: for each subject that
changed scanner, time is centred on the first session acquired on the new scanner. ADNI
releases exam dates, so a calendar-time cutoff per site can be used instead
(:func:`add_calendar_running_variable`).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_OASIS_RE = re.compile(r"^(OAS3\d{4})_[A-Za-z0-9]+_d(\d{4,5})$")


def parse_oasis_id(identifier: str) -> Tuple[str, int]:
    """Split an OASIS-3 experiment ID into ``(subject, days_from_entry)``.

    >>> parse_oasis_id("OAS30001_MR_d0129")
    ('OAS30001', 129)
    """
    m = _OASIS_RE.match(identifier.strip())
    if not m:
        raise ValueError(f"not an OASIS-3 experiment id: {identifier!r}")
    return m.group(1), int(m.group(2))


def build_session_table(
    mr_sessions: pd.DataFrame,
    id_col: str = "MR ID",
    scanner_col: str = "Scanner",
) -> pd.DataFrame:
    """Normalise an OASIS-3 MR session spreadsheet to ``subject, day, years, scanner``.

    Parameters
    ----------
    mr_sessions
        Table with one row per MR session. ``id_col`` holds ``OAS3xxxx_MR_dNNNN``;
        ``scanner_col`` holds the scanner model string.
    """
    parsed = mr_sessions[id_col].map(parse_oasis_id)
    out = pd.DataFrame(
        {
            "session_id": mr_sessions[id_col].values,
            "subject": [p[0] for p in parsed],
            "day": [p[1] for p in parsed],
            "scanner": mr_sessions[scanner_col].astype(str).str.strip().values,
        }
    )
    out["years"] = out["day"] / 365.25
    return out.sort_values(["subject", "day"]).reset_index(drop=True)


def find_transitions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Detect scanner changes within subjects.

    Returns one row per transition with ``subject, from_scanner, to_scanner, day_pre, day_post,
    cutoff_day, n_pre, n_post`` where ``cutoff_day`` is the day of the first session on the new
    scanner (the subject-anchored cutoff) and ``n_pre``/``n_post`` count sessions on the old
    scanner before and on the new scanner after that transition.
    """
    rows = []
    for subj, g in sessions.sort_values(["subject", "day"]).groupby("subject", sort=False):
        scanners = g["scanner"].tolist()
        days = g["day"].tolist()
        for i in range(1, len(scanners)):
            if scanners[i] != scanners[i - 1]:
                n_pre = sum(1 for k in range(i) if scanners[k] == scanners[i - 1])
                n_post = sum(1 for k in range(i, len(scanners)) if scanners[k] == scanners[i])
                rows.append(
                    {
                        "subject": subj,
                        "from_scanner": scanners[i - 1],
                        "to_scanner": scanners[i],
                        "day_pre": days[i - 1],
                        "day_post": days[i],
                        "cutoff_day": days[i],
                        "n_pre": n_pre,
                        "n_post": n_post,
                    }
                )
    cols = ["subject", "from_scanner", "to_scanner", "day_pre", "day_post", "cutoff_day", "n_pre", "n_post"]
    return pd.DataFrame(rows, columns=cols)


def add_event_time(
    sessions: pd.DataFrame,
    transitions: pd.DataFrame,
    pair: Tuple[str, str],
    min_pre: int = 1,
    min_post: int = 1,
) -> pd.DataFrame:
    """Attach the event-time running variable for one scanner pair.

    Keeps, for each subject with a ``pair[0] -> pair[1]`` transition, the sessions on either of
    the two scanners, and adds ``t_rel`` (years relative to the subject's cutoff, negative before)
    and ``post`` (1 on the new scanner). Subjects with fewer than ``min_pre``/``min_post`` sessions
    on the respective side are dropped. A subject with several transitions of the same pair keeps
    only the first.
    """
    tr = transitions[(transitions.from_scanner == pair[0]) & (transitions.to_scanner == pair[1])]
    tr = tr[(tr.n_pre >= min_pre) & (tr.n_post >= min_post)].drop_duplicates("subject", keep="first")
    cut = tr.set_index("subject")["cutoff_day"]
    df = sessions[sessions.subject.isin(cut.index) & sessions.scanner.isin(pair)].copy()
    df["cutoff_day"] = df["subject"].map(cut)
    df["t_rel"] = (df["day"] - df["cutoff_day"]) / 365.25
    df["post"] = (df["scanner"] == pair[1]).astype(int)
    # sessions on the old scanner *after* the cutoff (scanner went back and forth) are ambiguous
    df = df[~((df.post == 0) & (df.t_rel >= 0))]
    return df.reset_index(drop=True)


def find_paired_sessions(sessions: pd.DataFrame, max_days: int = 14) -> pd.DataFrame:
    """Traveling-subject pairs: same subject, two scanners, sessions ``<= max_days`` apart.

    OASIS-3 contains 69 participants scanned on the TIM Trio and the Biograph mMR within two
    weeks; this function recovers such pairs generically.
    """
    rows = []
    for subj, g in sessions.sort_values(["subject", "day"]).groupby("subject", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                gap = g.loc[j, "day"] - g.loc[i, "day"]
                if gap > max_days:
                    break
                if g.loc[i, "scanner"] != g.loc[j, "scanner"]:
                    rows.append(
                        {
                            "subject": subj,
                            "session_a": g.loc[i, "session_id"],
                            "scanner_a": g.loc[i, "scanner"],
                            "session_b": g.loc[j, "session_id"],
                            "scanner_b": g.loc[j, "scanner"],
                            "gap_days": int(gap),
                        }
                    )
    return pd.DataFrame(rows, columns=["subject", "session_a", "scanner_a", "session_b", "scanner_b", "gap_days"])


def add_calendar_running_variable(
    df: pd.DataFrame,
    date_col: str,
    site_col: str,
    cutoff_dates: dict,
) -> pd.DataFrame:
    """Calendar-time running variable (ADNI-style) with a site-specific cutoff date.

    ``cutoff_dates`` maps site id -> cutoff date (anything ``pd.Timestamp`` accepts). Rows whose
    site has no cutoff get ``NaN``.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    cut = out[site_col].map({k: pd.Timestamp(v) for k, v in cutoff_dates.items()})
    out["t_rel"] = (dates - cut).dt.days / 365.25
    out["post"] = (out["t_rel"] >= 0).astype(float)
    out.loc[out["t_rel"].isna(), "post"] = np.nan
    return out


def infer_site_cutoffs(df: pd.DataFrame, date_col: str, site_col: str, new_flag_col: str) -> dict:
    """First date at which each site produced a scan flagged as the new scanner/field strength."""
    d = df[df[new_flag_col].astype(bool)]
    return pd.to_datetime(d[date_col]).groupby(d[site_col]).min().to_dict()
