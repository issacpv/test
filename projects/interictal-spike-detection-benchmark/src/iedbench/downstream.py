"""Downstream consequences of detector choice: spike rates, SOZ ranking and vigilance stratification."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .events import Event


def spike_rate_per_channel(events: Sequence[Event], duration_s: float, channels: Sequence[str]) -> pd.Series:
    """Events per minute per channel (channels without events get 0)."""
    counts = pd.Series(0.0, index=list(channels))
    for e in events:
        if e.channel in counts.index:
            counts[e.channel] += 1
    return counts / (duration_s / 60.0)


def poisson_rate_ci(n_events: int, duration_min: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Garwood) confidence interval for a Poisson rate per minute."""
    lo = stats.chi2.ppf(alpha / 2, 2 * n_events) / 2 if n_events > 0 else 0.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (n_events + 1)) / 2
    return float(lo / duration_min), float(hi / duration_min)


def soz_ranking_auc(rates: pd.Series, soz_channels: Sequence[str]) -> float:
    """AUROC of spike rate for discriminating clinically annotated SOZ channels from the rest."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray([1 if c in set(soz_channels) else 0 for c in rates.index])
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(roc_auc_score(y, rates.values))


def rank_agreement(rates_a: pd.Series, rates_b: pd.Series) -> Dict[str, float]:
    """Spearman and Kendall agreement of channel rankings between two detectors."""
    idx = rates_a.index.intersection(rates_b.index)
    a, b = rates_a.loc[idx].values, rates_b.loc[idx].values
    if idx.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"spearman": float("nan"), "kendall": float("nan"), "n": int(idx.size)}
    return {"spearman": float(stats.spearmanr(a, b)[0]), "kendall": float(stats.kendalltau(a, b)[0]), "n": int(idx.size)}


def top_k_overlap(rates_a: pd.Series, rates_b: pd.Series, k: int = 3) -> float:
    """Jaccard overlap of the top-k channels of two detectors."""
    ta = set(rates_a.sort_values(ascending=False).index[:k])
    tb = set(rates_b.sort_values(ascending=False).index[:k])
    return len(ta & tb) / len(ta | tb) if ta | tb else float("nan")


def soz_multiverse(rates_by_detector: Dict[str, pd.Series], soz_channels: Sequence[str], k: int = 3) -> pd.DataFrame:
    """Per-detector SOZ AUROC plus pairwise top-k overlap and rank agreement summary."""
    names = list(rates_by_detector)
    rows = []
    for n in names:
        rows.append(dict(detector=n, soz_auc=soz_ranking_auc(rates_by_detector[n], soz_channels),
                         top_channel=rates_by_detector[n].idxmax()))
    df = pd.DataFrame(rows)
    ov, ra = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov.append(top_k_overlap(rates_by_detector[names[i]], rates_by_detector[names[j]], k))
            ra.append(rank_agreement(rates_by_detector[names[i]], rates_by_detector[names[j]])["spearman"])
    df.attrs["mean_topk_overlap"] = float(np.nanmean(ov)) if ov else float("nan")
    df.attrs["mean_spearman"] = float(np.nanmean(ra)) if ra else float("nan")
    df.attrs["auc_range"] = float(df["soz_auc"].max() - df["soz_auc"].min()) if df["soz_auc"].notna().any() else float("nan")
    return df


def vigilance_stratified_rates(events: Sequence[Event], stages: Sequence[Tuple[float, float, str]]) -> pd.DataFrame:
    """Events per minute per vigilance stage, given (start_s, end_s, stage) intervals."""
    t = np.asarray([e.t_s for e in events], float)
    rows = []
    for stage in sorted({s[2] for s in stages}):
        mins = sum(e - s for s, e, st in stages if st == stage) / 60.0
        n = int(sum(((t >= s) & (t < e)).sum() for s, e, st in stages if st == stage))
        lo, hi = poisson_rate_ci(n, mins) if mins > 0 else (np.nan, np.nan)
        rows.append(dict(stage=stage, minutes=mins, n_events=n, rate_per_min=n / mins if mins > 0 else np.nan,
                         ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def channel_permutation_null(rates: pd.Series, soz_channels: Sequence[str], n_perm: int = 1000, seed: int = 0) -> float:
    """p-value of the observed SOZ AUROC under random permutation of channel labels."""
    rng = np.random.default_rng(seed)
    obs = soz_ranking_auc(rates, soz_channels)
    if not np.isfinite(obs):
        return float("nan")
    k = len(set(soz_channels) & set(rates.index))
    null = np.empty(n_perm)
    idx = np.asarray(rates.index)
    for i in range(n_perm):
        null[i] = soz_ranking_auc(rates, rng.choice(idx, k, replace=False))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))
