"""Temporal validation harness, drift metrics, calibration and decision curves.

MIMIC-IV's ``anchor_year_group`` ("2008 - 2010", ..., "2020 - 2022") is the only
calendar-anchored time field, so temporal validation is done in *era blocks*:
train on earlier eras, test on later ones.  Patients never straddle splits
because the era is a patient-level attribute.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ERAS: tuple[str, ...] = ("2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022")


def era_index(era: pd.Series) -> pd.Series:
    """Map era strings to ordinal indices 0..4 (unknown -> -1)."""
    lut = {e: i for i, e in enumerate(ERAS)}
    return era.astype(str).str.strip().map(lut).fillna(-1).astype(int)


@dataclass
class TemporalSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_eras: tuple[str, ...]
    test_eras: tuple[str, ...]


def temporal_split(df: pd.DataFrame, train_eras: Sequence[str], test_eras: Sequence[str], era_col: str = "era") -> TemporalSplit:
    """Index arrays for an era-block split."""
    e = df[era_col].astype(str).str.strip()
    tr = np.where(e.isin(train_eras))[0]
    te = np.where(e.isin(test_eras))[0]
    return TemporalSplit(tr, te, tuple(train_eras), tuple(test_eras))


def rolling_origin_splits(df: pd.DataFrame, era_col: str = "era", min_train_eras: int = 2) -> Iterator[TemporalSplit]:
    """Yield train-on-earlier / test-on-next-era splits: (e0..ek-1 -> ek) for k >= min_train_eras."""
    present = [e for e in ERAS if (df[era_col].astype(str).str.strip() == e).any()]
    for k in range(min_train_eras, len(present)):
        yield temporal_split(df, present[:k], [present[k]], era_col)


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference and a current feature distribution (quantile bins on ref)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.histogram(ref, edges)[0] / max(len(ref), 1) + 1e-6
    c = np.histogram(cur, edges)[0] / max(len(cur), 1) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(df: pd.DataFrame, features: Sequence[str], ref_eras: Sequence[str], era_col: str = "era",
                 outcomes: Sequence[str] = ("hospitalization", "critical_outcome", "revisit_72h")) -> pd.DataFrame:
    """PSI and KS statistic per feature and prevalence per outcome, for each era vs the reference eras."""
    e = df[era_col].astype(str).str.strip()
    ref = df[e.isin(ref_eras)]
    rows = []
    for era in [x for x in ERAS if (e == x).any()]:
        cur = df[e == era]
        row = {"era": era, "n": len(cur)}
        for f in features:
            a, b = pd.to_numeric(ref[f], errors="coerce").to_numpy(), pd.to_numeric(cur[f], errors="coerce").to_numpy()
            row[f"psi_{f}"] = population_stability_index(a, b)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            row[f"ks_{f}"] = float(stats.ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan
        for o in outcomes:
            if o in cur:
                row[f"prev_{o}"] = float(cur[o].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Discrimination / calibration / decision curves
# ----------------------------------------------------------------------------
def discrimination(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if y.sum() in (0, len(y)):
        return {"auroc": np.nan, "auprc": np.nan, "prevalence": float(y.mean())}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)), "prevalence": float(y.mean())}


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: regress y on logit(p). Slope 1 / intercept 0 = perfect; intercept drift signals prevalence shift."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    X = np.c_[np.ones_like(lp), lp]
    beta = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1]), float(beta[0])


def calibration_in_the_large(y: np.ndarray, p: np.ndarray) -> float:
    """Observed minus expected event rate (positive = under-prediction)."""
    return float(np.mean(y) - np.mean(p))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned observed vs predicted rates for reliability plots."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = [{"bin": b, "n": int((idx == b).sum()), "pred_mean": float(p[idx == b].mean()), "obs_rate": float(y[idx == b].mean())}
            for b in range(n_bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: Sequence[float]) -> pd.DataFrame:
    """Decision curve: net benefit of 'treat if p >= t' vs treat-all and treat-none, per threshold probability."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)) / n
        fp = np.sum(pred & (y == 0)) / n
        odds = t / (1 - t)
        rows.append({"threshold": t, "net_benefit_model": tp - fp * odds,
                     "net_benefit_all": prev - (1 - prev) * odds, "net_benefit_none": 0.0})
    return pd.DataFrame(rows)


def esi_as_score(acuity: np.ndarray) -> np.ndarray:
    """Turn ESI (1 = most urgent) into an increasing risk score in [0, 1] for AUROC / decision-curve comparison."""
    a = np.asarray(acuity, float)
    return (5 - a) / 4.0


def evaluate_split(y_train: np.ndarray, p_train: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
                   thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.3)) -> dict[str, float]:
    """One-row summary of discrimination + calibration in train and test, and net benefit in test."""
    out = {f"train_{k}": v for k, v in discrimination(y_train, p_train).items()}
    out.update({f"test_{k}": v for k, v in discrimination(y_test, p_test).items()})
    slope, intercept = calibration_slope_intercept(y_test, p_test)
    out.update({"test_cal_slope": slope, "test_cal_intercept": intercept,
                "test_citl": calibration_in_the_large(y_test, p_test), "test_ece": expected_calibration_error(y_test, p_test)})
    nb = net_benefit(y_test, p_test, thresholds)
    for _, r in nb.iterrows():
        out[f"test_nb@{r['threshold']:.2f}"] = float(r["net_benefit_model"])
    return out


def run_temporal_benchmark(df: pd.DataFrame, fit_predict: Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]],
                           outcome: str, era_col: str = "era", min_train_eras: int = 2) -> pd.DataFrame:
    """Apply ``fit_predict(train_df, test_df) -> (p_train, p_test)`` across rolling-origin era splits."""
    rows = []
    for sp in rolling_origin_splits(df, era_col, min_train_eras):
        tr, te = df.iloc[sp.train_idx], df.iloc[sp.test_idx]
        p_tr, p_te = fit_predict(tr, te)
        row = {"train_eras": "+".join(sp.train_eras), "test_era": sp.test_eras[0], "n_train": len(tr), "n_test": len(te)}
        row.update(evaluate_split(tr[outcome].to_numpy(), p_tr, te[outcome].to_numpy(), p_te))
        rows.append(row)
    return pd.DataFrame(rows)
