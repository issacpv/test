"""Exposure denominators for FAERS from public prescribing data.

Two open denominator sources are supported:

1. **Medicare Part D "Spending by Drug"** (data.cms.gov). One row per
   brand/generic with yearly ``Tot_Clms_<YYYY>``, ``Tot_Benes_<YYYY>`` and
   ``Tot_Dsg_Unts_<YYYY>`` columns for the last 5 years. Sex-pooled, age >= 65
   dominated (plus disabled beneficiaries). Good for age/secular trends and as a
   robustness denominator.
2. **MEPS Household Component Prescribed Medicines files** (AHRQ, e.g. HC-248A
   for 2023) merged with the Full-Year Consolidated file for ``SEX`` and ``AGE``.
   Event-level with survey weights (``PERWT23F`` etc.), so we can estimate the
   number of US civilian non-institutionalised persons with >= 1 fill of a drug
   by sex and age band. This is the sex-stratified denominator.

The unit of the merged table is (drug, year, [sex, age_band]) with a FAERS
report count and an exposure count; :func:`reporting_rate_ratio` then gives the
female:male ratio of *reports per exposed person* with a Poisson CI, which is
the quantity FAERS alone cannot provide.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Salt / ester / hydrate qualifiers commonly appended to generic names in Part D
# and openFDA ``generic_name`` (e.g. "ATORVASTATIN CALCIUM", "METFORMIN HCL").
_SALT_TOKENS = {
    "HCL", "HYDROCHLORIDE", "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "MALEATE",
    "MESYLATE", "TARTRATE", "BITARTRATE", "SUCCINATE", "FUMARATE", "CITRATE", "ACETATE",
    "SULFATE", "PHOSPHATE", "BESYLATE", "TOSYLATE", "BROMIDE", "CHLORIDE", "NITRATE",
    "DIHYDRATE", "MONOHYDRATE", "ANHYDROUS", "HYDROBROMIDE", "LACTATE", "PAMOATE",
    "DECANOATE", "VALERATE", "PROPIONATE", "DIPROPIONATE", "ER", "XR", "SR", "DR", "ODT",
    "HYCLATE", "MONONITRATE", "TROMETHAMINE", "ARGININE", "MEGLUMINE", "ESYLATE",
}
_PUNCT_RE = re.compile(r"[^A-Z0-9/ ]+")


def normalize_drug_name(name: str, keep_combinations: bool = True) -> str:
    """Upper-case, strip punctuation and salt/ester qualifiers.

    ``"Atorvastatin Calcium"`` -> ``"ATORVASTATIN"``;
    ``"Sacubitril/Valsartan"`` -> ``"SACUBITRIL/VALSARTAN"`` (combinations are kept
    in canonical alphabetical order so that "A/B" == "B/A").
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).upper().strip()
    s = s.replace("-", " ").replace(",", " ")
    s = _PUNCT_RE.sub(" ", s)
    parts = [p.strip() for p in s.split("/")] if "/" in s and keep_combinations else [s]
    cleaned: List[str] = []
    for p in parts:
        toks = [t for t in p.split() if t and t not in _SALT_TOKENS]
        if toks:
            cleaned.append(" ".join(toks))
    return "/".join(sorted(cleaned))


def load_partd_spending_by_drug(path: str) -> pd.DataFrame:
    """Load the CMS "Medicare Part D Spending by Drug" CSV into long format.

    Returns columns ``drug_norm, brand_name, generic_name, year, total_claims,
    total_benes, total_dosage_units`` (one row per generic x year). Rows with
    suppressed beneficiary counts (CMS suppresses cells < 11) come back as NaN.
    """
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    gen_col = next((c for c in raw.columns if c.lower().startswith("gnrc_name")), None)
    brand_col = next((c for c in raw.columns if c.lower().startswith("brnd_name")), None)
    if gen_col is None:
        raise ValueError("could not find Gnrc_Name column in Part D file")
    year_cols = {}
    for c in raw.columns:
        m = re.match(r"(Tot_Clms|Tot_Benes|Tot_Dsg_Unts)_(\d{4})$", c)
        if m:
            year_cols.setdefault(m.group(2), {})[m.group(1)] = c
    rows = []
    for _, r in raw.iterrows():
        for year, cols in year_cols.items():
            rows.append(
                {
                    "generic_name": r[gen_col],
                    "brand_name": r[brand_col] if brand_col else None,
                    "year": int(year),
                    "total_claims": pd.to_numeric(r.get(cols.get("Tot_Clms")), errors="coerce"),
                    "total_benes": pd.to_numeric(r.get(cols.get("Tot_Benes")), errors="coerce"),
                    "total_dosage_units": pd.to_numeric(r.get(cols.get("Tot_Dsg_Unts")), errors="coerce"),
                }
            )
    long = pd.DataFrame(rows)
    long["drug_norm"] = long["generic_name"].map(normalize_drug_name)
    # several manufacturers/brands per generic -> sum to generic level
    agg = (
        long.groupby(["drug_norm", "year"], as_index=False)[["total_claims", "total_benes", "total_dosage_units"]]
        .sum(min_count=1)
    )
    return agg


def age_band(age: pd.Series, edges: Sequence[int] = (0, 18, 45, 65, 200)) -> pd.Series:
    """Cut ages into labelled bands like ``"18-44"``; last band is ``"65+"``."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo}-{hi - 1}" if hi < 150 else f"{lo}+")
    return pd.cut(pd.to_numeric(age, errors="coerce"), bins=list(edges), right=False, labels=labels).astype(str)


def aggregate_meps_pmed(
    pmed: pd.DataFrame,
    persons: pd.DataFrame,
    year: int,
    drug_col: str = "RXDRGNAM",
    person_id_col: str = "DUPERSID",
    weight_col: Optional[str] = None,
    sex_col: str = "SEX",
    age_col: Optional[str] = None,
    age_edges: Sequence[int] = (0, 18, 45, 65, 200),
) -> pd.DataFrame:
    """Weighted number of persons with >= 1 fill per (drug, sex, age band).

    ``pmed`` is the Prescribed Medicines event file; ``persons`` the Full-Year
    Consolidated file (contains ``SEX`` 1=male 2=female, ``AGE<yy>X`` and the
    person weight ``PERWT<yy>F``). Column names default to the MEPS conventions
    for ``year`` when not supplied.
    """
    yy = f"{year % 100:02d}"
    weight_col = weight_col or f"PERWT{yy}F"
    age_col = age_col or f"AGE{yy}X"
    keep = [person_id_col, sex_col, weight_col] + ([age_col] if age_col in persons.columns else [])
    p = persons[keep].drop_duplicates(person_id_col)
    d = pmed[[person_id_col, drug_col]].dropna().copy()
    d["drug_norm"] = d[drug_col].map(normalize_drug_name)
    d = d.drop_duplicates([person_id_col, "drug_norm"])  # persons, not fills
    m = d.merge(p, on=person_id_col, how="inner")
    m["sex"] = m[sex_col].map({1: "male", 2: "female", "1": "male", "2": "female"}).fillna("unknown")
    m["age_band"] = age_band(m[age_col], age_edges) if age_col in m.columns else "all"
    out = (
        m.groupby(["drug_norm", "sex", "age_band"], as_index=False)
        .agg(exposed_persons=(weight_col, "sum"), unweighted_n=(person_id_col, "nunique"))
    )
    out["year"] = int(year)
    return out


def merge_denominators(
    faers_counts: pd.DataFrame,
    denominators: pd.DataFrame,
    on: Sequence[str] = ("drug_norm", "year", "sex"),
    exposure_col: str = "exposed_persons",
    per: float = 10_000.0,
) -> pd.DataFrame:
    """Join FAERS report counts with exposure and compute reports per ``per`` exposed.

    ``faers_counts`` must contain the ``on`` keys and ``reports``; the result
    adds ``rate`` and an exact Poisson 95% CI (``rate_lo``, ``rate_hi``).
    """
    keys = list(on)
    m = faers_counts.merge(denominators, on=keys, how="inner", validate="one_to_one")
    n = m["reports"].astype(float).values
    e = m[exposure_col].astype(float).values
    with np.errstate(divide="ignore", invalid="ignore"):
        m["rate"] = per * n / e
        lo = stats.chi2.ppf(0.025, 2 * n) / 2
        hi = stats.chi2.ppf(0.975, 2 * (n + 1)) / 2
        m["rate_lo"] = per * np.where(n > 0, lo, 0.0) / e
        m["rate_hi"] = per * hi / e
    return m


def reporting_rate_ratio(
    merged: pd.DataFrame,
    group_col: str = "sex",
    numerator_level: str = "female",
    denominator_level: str = "male",
    keys: Sequence[str] = ("drug_norm", "year"),
    exposure_col: str = "exposed_persons",
) -> pd.DataFrame:
    """Female:male ratio of reports-per-exposed with Wald CI on the log scale.

    RRR = (n_f / E_f) / (n_m / E_m); SE(log RRR) ~ sqrt(1/n_f + 1/n_m) treating
    the exposure denominators as fixed (survey-design variance can be added via
    replicate weights if needed). Also returns the *crude* FAERS ratio n_f / n_m
    so the effect of the denominator can be read off directly.
    """
    keys = list(keys)
    f = merged[merged[group_col] == numerator_level].set_index(keys)
    m = merged[merged[group_col] == denominator_level].set_index(keys)
    common = f.index.intersection(m.index)
    f = f.loc[common]
    m = m.loc[common]
    nf, nm = f["reports"].astype(float), m["reports"].astype(float)
    ef, em = f[exposure_col].astype(float), m[exposure_col].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rrr = (nf / ef) / (nm / em)
        se = np.sqrt(1.0 / nf.clip(lower=0.5) + 1.0 / nm.clip(lower=0.5))
        out = pd.DataFrame(
            {
                "reports_f": nf,
                "reports_m": nm,
                "exposed_f": ef,
                "exposed_m": em,
                "crude_ratio": nf / nm,
                "exposure_ratio": ef / em,
                "rrr": rrr,
                "rrr_lo": np.exp(np.log(rrr) - 1.96 * se),
                "rrr_hi": np.exp(np.log(rrr) + 1.96 * se),
            }
        )
    return out.reset_index()


def faers_counts_by_sex_year(flat: pd.DataFrame, drug_col: str = "suspect_generic_names") -> pd.DataFrame:
    """Explode a flattened FAERS frame (see ``openfda_client.records_to_frame``)
    into (drug_norm, year, sex) report counts."""
    df = flat[["safetyreportid", "receivedate", "sex", drug_col]].explode(drug_col).dropna(subset=[drug_col])
    df["drug_norm"] = df[drug_col].map(normalize_drug_name)
    df["year"] = pd.to_datetime(df["receivedate"], format="%Y%m%d", errors="coerce").dt.year
    df = df.drop_duplicates(["safetyreportid", "drug_norm"])
    return df.groupby(["drug_norm", "year", "sex"], as_index=False).agg(reports=("safetyreportid", "nunique"))
