"""Exposure denominators and sex-specific reporting rates.

Spontaneous-report proportions cannot separate "women experience more
events" from "more women take the drug" and "women report more". Two open
US survey sources give survey-weighted, sex-specific numbers of users:

* MEPS Prescribed Medicines files (HC-xxxA), one row per fill with
  ``RXDRGNAM`` / ``RXNDC`` and ``DUPERSID`` linking to the Full-Year
  Consolidated file (``SEX``, ``AGELAST``, ``PERWTyyF``). Open download:
  https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp
* NHANES RXQ_RX (prescription medications, generic names via the
  Multum lexicon) with demographics in DEMO and weights ``WTINTPRP`` /
  ``WTINT2YR``. Open: https://wwwn.cdc.gov/nchs/nhanes/

Functions here compute weighted users by sex, Poisson reporting rates per
10 000 users, sex rate ratios, and the "excess female reporting" statistic:
observed female share of reports vs the share expected from exposure alone.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

GLP1_NAME_RE = re.compile(r"(semaglutide|tirzepatide|liraglutide|dulaglutide|exenatide|lixisenatide|albiglutide|ozempic|wegovy|rybelsus|mounjaro|zepbound|victoza|saxenda|trulicity|byetta|bydureon|adlyxin)", re.I)


def poisson_rate(count: float, exposure: float, per: float = 10_000.0) -> Dict[str, float]:
    """Rate per ``per`` exposed with exact (Garwood) 95% CI."""
    k = float(count)
    lo = 0.5 * stats.chi2.ppf(0.025, 2 * k) if k > 0 else 0.0
    hi = 0.5 * stats.chi2.ppf(0.975, 2 * (k + 1))
    f = per / exposure if exposure > 0 else float("nan")
    return {"rate": k * f, "rate_lo": lo * f, "rate_hi": hi * f, "count": k, "exposure": float(exposure)}


def rate_ratio(c1: float, e1: float, c2: float, e2: float) -> Dict[str, float]:
    """Rate ratio (group 1 / group 2) with log-normal 95% CI and p-value."""
    if min(c1, c2) == 0:
        c1, c2 = c1 + 0.5, c2 + 0.5
    rr = (c1 / e1) / (c2 / e2)
    se = float(np.sqrt(1 / c1 + 1 / c2))
    z = np.log(rr) / se
    return {"rr": float(rr), "rr_lo": float(np.exp(np.log(rr) - 1.96 * se)), "rr_hi": float(np.exp(np.log(rr) + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(z)))}


def sex_reporting_rates(report_counts: Dict[str, float], users: Dict[str, float], per: float = 10_000.0) -> pd.DataFrame:
    """Reports per ``per`` users by sex, plus the female/male rate ratio row."""
    rows = []
    for sex in ("female", "male"):
        r = poisson_rate(report_counts.get(sex, 0), users.get(sex, 0), per)
        rows.append({"sex": sex, **r})
    rr = rate_ratio(report_counts.get("female", 0), users.get("female", 1), report_counts.get("male", 0), users.get("male", 1))
    rows.append({"sex": "female/male", "rate": rr["rr"], "rate_lo": rr["rr_lo"], "rate_hi": rr["rr_hi"], "count": np.nan, "exposure": np.nan, "p": rr["p"]})
    return pd.DataFrame(rows)


def excess_female_reporting(n_female_reports: int, n_male_reports: int, users_female: float, users_male: float) -> Dict[str, float]:
    """Observed vs exposure-expected female share of reports.

    Under equal per-user reporting propensity the expected female share is
    users_female / (users_female + users_male). Returns the observed share,
    expected share, their ratio (the "reporting propensity ratio") and an
    exact binomial p-value.
    """
    n = n_female_reports + n_male_reports
    expected = users_female / (users_female + users_male) if (users_female + users_male) > 0 else float("nan")
    observed = n_female_reports / n if n else float("nan")
    p = float(stats.binomtest(n_female_reports, n, expected).pvalue) if n and 0 < expected < 1 else float("nan")
    return {"observed_female_share": observed, "expected_female_share": expected, "propensity_ratio": observed / expected if expected else float("nan"), "p_binomial": p, "n_reports": n}


def weighted_users_by_sex(persons: pd.DataFrame, sex_col: str, weight_col: str, user_col: str, female_code: object = 2, male_code: object = 1) -> Dict[str, float]:
    """Survey-weighted number of users by sex from a person-level frame.

    ``user_col`` is a 0/1 flag (person had >= 1 fill of the drug class in the
    survey year). Weights are summed, so the result is a population count.
    """
    u = persons[persons[user_col].astype(bool)]
    return {
        "female": float(u.loc[u[sex_col] == female_code, weight_col].sum()),
        "male": float(u.loc[u[sex_col] == male_code, weight_col].sum()),
    }


def flag_glp1_fills(rx: pd.DataFrame, name_cols: Sequence[str] = ("RXDRGNAM", "RXNAME"), pattern: re.Pattern = GLP1_NAME_RE) -> pd.Series:
    """Boolean mask of prescription rows matching GLP-1 names (MEPS/NHANES)."""
    mask = pd.Series(False, index=rx.index)
    for col in name_cols:
        if col in rx.columns:
            mask |= rx[col].astype(str).str.contains(pattern, na=False)
    return mask


def users_from_fills(rx: pd.DataFrame, persons: pd.DataFrame, id_col: str, fill_mask: pd.Series, sex_col: str, weight_col: str) -> Dict[str, float]:
    """Join flagged fills to persons and return weighted users by sex."""
    ids = set(rx.loc[fill_mask, id_col])
    p = persons.copy()
    p["_user"] = p[id_col].isin(ids).astype(int)
    return weighted_users_by_sex(p, sex_col, weight_col, "_user")


def standardise_by_age(report_counts: pd.DataFrame, users: pd.DataFrame, per: float = 10_000.0) -> pd.DataFrame:
    """Direct age-standardised reporting rate per sex.

    Both inputs have columns ``sex, age_band, n`` (reports) / ``sex, age_band,
    users``; the standard population is the pooled user age distribution.
    """
    std = users.groupby("age_band")["users"].sum()
    std = std / std.sum()
    out = []
    for sex, g in report_counts.groupby("sex"):
        u = users[users["sex"] == sex].set_index("age_band")["users"]
        rate = 0.0
        for band, w in std.items():
            n = float(g.loc[g["age_band"] == band, "n"].sum())
            denom = float(u.get(band, 0.0))
            rate += w * (n / denom * per if denom > 0 else 0.0)
        out.append({"sex": sex, "std_rate": rate})
    return pd.DataFrame(out)
