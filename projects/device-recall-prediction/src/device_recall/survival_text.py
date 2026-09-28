"""Landmark time-to-recall dataset, MAUDE text features and Cox models.

Design
------
* Unit of analysis: a cleared/approved device (K or P number) or, when reports
  cannot be linked to a submission, a (product code, firm) family.
* Time origin: clearance date + ``landmark_months``. Features are computed from
  MAUDE reports received *within* the landmark window only (early-warning
  setting); follow-up runs from the landmark to the first recall
  (any class, or Class I/II) or administrative censoring at ``study_end``.
  Devices recalled inside the landmark window are excluded (immortal-time
  guard) and counted separately.
* Text: TF-IDF (1-2 grams) on concatenated event narratives reduced with
  truncated SVD to ``k`` components; optional sentence-transformer embeddings.
* Model: Cox proportional hazards via ``lifelines`` when installed, otherwise
  ``statsmodels`` ``PHReg``. Validation is temporal: train on devices cleared
  before a split date, evaluate Harrell's C on devices cleared after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional
    from lifelines import CoxPHFitter  # type: ignore

    _HAS_LIFELINES = True
except Exception:  # pragma: no cover
    CoxPHFitter = None  # type: ignore
    _HAS_LIFELINES = False


# ----------------------------------------------------------------- dataset
def build_survival_table(
    devices: pd.DataFrame,
    recalls: pd.DataFrame,
    study_end: str,
    landmark_months: int = 12,
    recall_classes: Sequence[str] = ("Class I", "Class II"),
    id_col: str = "k_number",
    date_col: str = "decision_date",
) -> pd.DataFrame:
    """Return one row per device with ``duration_months``, ``event`` and flags.

    ``recalls`` has ``submission_number, recall_date, recall_class``.
    """
    dev = devices.dropna(subset=[id_col, date_col]).copy()
    dev[id_col] = dev[id_col].astype(str).str.upper()
    dev[date_col] = pd.to_datetime(dev[date_col])
    rec = recalls.copy()
    rec["submission_number"] = rec["submission_number"].astype(str).str.upper()
    rec = rec[rec["recall_class"].isin(list(recall_classes))]
    rec["recall_date"] = pd.to_datetime(rec["recall_date"])
    first = rec.groupby("submission_number")["recall_date"].min().rename("first_recall")
    dev = dev.merge(first, left_on=id_col, right_index=True, how="left")
    end = pd.Timestamp(study_end)
    dev["landmark_date"] = dev[date_col] + pd.DateOffset(months=int(landmark_months))
    dev["recalled_before_landmark"] = dev["first_recall"].notna() & (dev["first_recall"] <= dev["landmark_date"])
    dev = dev[dev["landmark_date"] < end].copy()  # need follow-up after landmark
    out = dev[~dev["recalled_before_landmark"]].copy()
    out["event"] = (out["first_recall"].notna() & (out["first_recall"] <= end)).astype(int)
    stop = out["first_recall"].where(out["event"] == 1, end)
    out["duration_months"] = ((stop - out["landmark_date"]).dt.days / 30.4375).clip(lower=1e-3)
    return out


def maude_landmark_features(
    maude: pd.DataFrame,
    devices: pd.DataFrame,
    landmark_months: int = 12,
    id_col: str = "k_number",
    date_col: str = "decision_date",
) -> pd.DataFrame:
    """Early-warning features from MAUDE reports within the landmark window.

    ``maude`` needs ``k_number`` (from entity linking), ``date_received``,
    ``event_type`` (Death / Injury / Malfunction / Other), ``product_problems``
    (list) and ``narrative``. Returns counts, rates, problem diversity, a
    within-window slope of monthly counts and the concatenated narrative text.
    """
    dev = devices[[id_col, date_col]].dropna().copy()
    dev[id_col] = dev[id_col].astype(str).str.upper()
    dev["landmark_date"] = pd.to_datetime(dev[date_col]) + pd.DateOffset(months=int(landmark_months))
    m = maude.dropna(subset=[id_col]).copy()
    m[id_col] = m[id_col].astype(str).str.upper()
    m = m.merge(dev, on=id_col, how="inner")
    m["date_received"] = pd.to_datetime(m["date_received"])
    m = m[(m["date_received"] >= pd.to_datetime(m[date_col])) & (m["date_received"] < m["landmark_date"])]
    rows = []
    for k, g in m.groupby(id_col):
        et = g["event_type"].fillna("").str.lower()
        months = ((g["date_received"] - pd.to_datetime(g[date_col])).dt.days / 30.4375).astype(int)
        counts = months.value_counts().reindex(range(int(landmark_months)), fill_value=0).values
        slope = float(np.polyfit(np.arange(len(counts)), counts, 1)[0]) if len(counts) > 1 else 0.0
        probs = set()
        for p in g["product_problems"]:
            if isinstance(p, (list, tuple)):
                probs.update(p)
        rows.append(
            {
                id_col: k,
                "n_reports": len(g),
                "n_deaths": int((et == "death").sum()),
                "n_injuries": int((et == "injury").sum()),
                "n_malfunctions": int((et == "malfunction").sum()),
                "reports_per_month": len(g) / float(landmark_months),
                "n_distinct_problems": len(probs),
                "report_slope": slope,
                "narrative": " ".join(g["narrative"].fillna("").astype(str)),
            }
        )
    feat = pd.DataFrame(rows, columns=[id_col, "n_reports", "n_deaths", "n_injuries", "n_malfunctions", "reports_per_month", "n_distinct_problems", "report_slope", "narrative"])
    out = dev[[id_col]].merge(feat, on=id_col, how="left")
    num = ["n_reports", "n_deaths", "n_injuries", "n_malfunctions", "reports_per_month", "n_distinct_problems", "report_slope"]
    out[num] = out[num].fillna(0)
    out["narrative"] = out["narrative"].fillna("")
    out["log1p_reports"] = np.log1p(out["n_reports"])
    return out


# -------------------------------------------------------------- text feats
def tfidf_features(
    texts: Sequence[str],
    n_components: int = 20,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    fit_mask: Optional[np.ndarray] = None,
    random_state: int = 0,
):
    """TF-IDF -> truncated SVD components (dense) for Cox regression.

    ``fit_mask`` restricts vectoriser/SVD fitting to training rows (temporal
    validation without leakage); all rows are transformed.
    Returns ``(components ndarray, vectorizer, svd)``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t if isinstance(t, str) else "" for t in texts]
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2, stop_words="english", sublinear_tf=True)
    idx = np.arange(len(texts)) if fit_mask is None else np.where(fit_mask)[0]
    fit_texts = [texts[i] for i in idx] or [""]
    vec.fit(fit_texts)
    X = vec.transform(texts)
    k = int(min(n_components, max(1, X.shape[1] - 1)))
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    svd.fit(X[idx] if len(idx) else X)
    return svd.transform(X), vec, svd


def sentence_embeddings(texts: Sequence[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:  # pragma: no cover
    """Optional dense embeddings (requires ``sentence-transformers``)."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode([t or "" for t in texts], batch_size=64, show_progress_bar=False))


# -------------------------------------------------------------------- Cox
@dataclass
class CoxResult:
    backend: str
    summary: pd.DataFrame  # index = covariate; columns hr, hr_lo, hr_hi, p
    concordance_train: float
    _predict: object

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Relative risk score (higher = earlier recall expected)."""
        return np.asarray(self._predict(X)).ravel()


def concordance_index(durations: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: P(risk_i > risk_j | T_i < T_j, event_i = 1), ties = 0.5."""
    t = np.asarray(durations, float)
    e = np.asarray(events, int)
    r = np.asarray(risk, float)
    num = den = 0.0
    for i in np.where(e == 1)[0]:
        comp = t > t[i]
        den += comp.sum()
        num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return float(num / den) if den > 0 else float("nan")


def fit_cox(
    df: pd.DataFrame,
    covariates: Sequence[str],
    duration_col: str = "duration_months",
    event_col: str = "event",
    penalizer: float = 0.01,
    strata: Optional[str] = None,
) -> CoxResult:
    """Cox PH with lifelines (preferred) or statsmodels PHReg fallback."""
    cols = list(covariates)
    data = df[cols + [duration_col, event_col] + ([strata] if strata else [])].dropna().copy()
    if _HAS_LIFELINES:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(data, duration_col=duration_col, event_col=event_col, strata=strata)
        s = cph.summary
        summary = pd.DataFrame(
            {"hr": s["exp(coef)"], "hr_lo": s["exp(coef) lower 95%"], "hr_hi": s["exp(coef) upper 95%"], "p": s["p"]}
        )
        c = float(cph.concordance_index_)
        return CoxResult("lifelines", summary, c, lambda X: cph.predict_partial_hazard(X[cols]))
    from statsmodels.duration.hazard_regression import PHReg

    X = data[cols].astype(float)
    kw = {}
    if strata:
        kw["strata"] = data[strata].values
    model = PHReg(data[duration_col].values, X, status=data[event_col].values, ties="efron", **kw)
    res = model.fit_regularized(alpha=penalizer) if penalizer > 0 else model.fit()
    params = pd.Series(np.asarray(res.params).ravel(), index=cols)
    try:
        bse = pd.Series(np.asarray(res.bse).ravel(), index=cols)
    except Exception:  # regularized fits have no bse
        bse = pd.Series(np.nan, index=cols)
    from scipy import stats

    summary = pd.DataFrame(
        {
            "hr": np.exp(params),
            "hr_lo": np.exp(params - 1.96 * bse),
            "hr_hi": np.exp(params + 1.96 * bse),
            "p": 2 * stats.norm.sf(np.abs(params / bse)),
        }
    )
    risk_train = X.values @ params.values
    c = concordance_index(data[duration_col].values, data[event_col].values, risk_train)
    return CoxResult("statsmodels", summary, c, lambda Xn: np.exp(Xn[cols].astype(float).values @ params.values))


def temporal_split_evaluate(
    df: pd.DataFrame,
    covariates: Sequence[str],
    split_date: str,
    date_col: str = "decision_date",
    duration_col: str = "duration_months",
    event_col: str = "event",
    **fit_kw,
) -> Dict[str, object]:
    """Train on devices cleared before ``split_date``; test C-index on the rest."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    train = d[d[date_col] < pd.Timestamp(split_date)]
    test = d[d[date_col] >= pd.Timestamp(split_date)]
    res = fit_cox(train, covariates, duration_col, event_col, **fit_kw)
    risk = res.predict_risk(test)
    c_test = concordance_index(test[duration_col].values, test[event_col].values, risk)
    return {"result": res, "c_train": res.concordance_train, "c_test": c_test, "n_train": len(train), "n_test": len(test), "events_test": int(test[event_col].sum())}
