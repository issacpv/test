"""Subgroup-specific label-noise injection and its effect on fairness metrics and AUC.

Two mechanisms by which report-derived label noise distorts subgroup metrics are simulated:

1. Evaluation-level (instance-dependent noise). Non-mention false negatives are not random: subtle
   findings are the ones left out of reports. If the non-mention rate differs by subgroup, the
   *noisy-positive* sets of the subgroups differ in severity mix, and an identical classifier shows
   different measured FNRs. Purely class-conditional FN noise (independent of severity) does *not*
   change the FNR measured among noisy positives; FP noise does (negatives enter the positive set).
2. Training-level. A model trained on differentially noisy labels learns to call subtle findings
   negative in the high-noise subgroup; evaluated against expert labels it shows a genuine FNR gap.

Functions
---------
inject_noise                 – flip labels with subgroup FN/FP rates, optionally severity-dependent.
estimate_noise_rates         – confident-learning style class-conditional noise matrix (Northcutt et al., 2021).
fnr_gap                      – FNR per subgroup and max-min gap at a threshold.
simulate_fairness_gap        – mechanism 1 with a fixed, fair latent-score classifier.
simulate_training_under_noise – mechanism 2 with a logistic model trained on noisy labels.
rank_agreement               – Kendall tau of model rankings under report vs expert labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def inject_noise(y_true: np.ndarray, group: np.ndarray, fnr_by_group: dict, fpr_by_group: dict,
                 rng: np.random.Generator | None = None, severity: np.ndarray | None = None,
                 severity_dependence: float = 0.0) -> np.ndarray:
    """Flip labels: positives → 0 with (group-average) prob ``fnr[g]``, negatives → 1 with ``fpr[g]``.

    With ``severity`` (higher = more conspicuous finding) and ``severity_dependence > 0`` the FN
    probability of a positive is weighted by ``2·sigmoid(-k·z)`` (z = standardized severity among the
    group's positives) and renormalised so the group's *average* FN rate stays ``fnr[g]``.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y_true, int)
    g = np.asarray(group)
    u = rng.random(y.size)
    out = y.copy()
    for lvl in np.unique(g):
        m = g == lvl
        fnr = float(fnr_by_group.get(lvl, 0.0))
        fpr = float(fpr_by_group.get(lvl, 0.0))
        pos = m & (y == 1)
        p_fn = np.full(y.size, fnr)
        if severity is not None and severity_dependence > 0 and pos.sum() > 1:
            s = np.asarray(severity, float)
            z = (s[pos] - s[pos].mean()) / (s[pos].std() + 1e-12)
            w = 2.0 * _sigmoid(-severity_dependence * z)
            w = w / w.mean()
            p_fn[pos] = np.clip(fnr * w, 0.0, 1.0)
        out[pos & (u < p_fn)] = 0
        out[m & (y == 0) & (u < fpr)] = 1
    return out


def estimate_noise_rates(probs: np.ndarray, noisy_labels: np.ndarray, group: np.ndarray | None = None) -> dict:
    """Confident-joint estimate of P(noisy = j | true = i) for binary labels.

    Thresholds are the mean predicted probability of each noisy class; an example is 'confidently'
    class 1 if p ≥ t1 and confidently class 0 if (1-p) ≥ t0. Returns the estimated noise matrix
    (rows = true class, cols = noisy class) overall and per group.
    """
    p, y = np.asarray(probs, float), np.asarray(noisy_labels, int)

    def one(mask: np.ndarray) -> np.ndarray:
        pp, yy = p[mask], y[mask]
        t1 = pp[yy == 1].mean() if np.any(yy == 1) else 0.5
        t0 = (1 - pp[yy == 0]).mean() if np.any(yy == 0) else 0.5
        conf1 = pp >= t1
        conf0 = (1 - pp) >= t0
        C = np.zeros((2, 2))
        C[1, 1] = np.sum(conf1 & (yy == 1))
        C[1, 0] = np.sum(conf1 & (yy == 0))
        C[0, 0] = np.sum(conf0 & (yy == 0))
        C[0, 1] = np.sum(conf0 & (yy == 1))
        rows = C.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(rows > 0, C / rows, np.nan)

    out = {"overall": one(np.ones(p.size, bool))}
    if group is not None:
        g = np.asarray(group)
        out["by_group"] = {lvl: one(g == lvl) for lvl in np.unique(g)}
    return out


def fnr_gap(scores: np.ndarray, labels: np.ndarray, group: np.ndarray, threshold: float) -> dict:
    """FNR per group at a fixed threshold and the max-min gap; labels may be true or noisy."""
    s, y, g = np.asarray(scores, float), np.asarray(labels, int), np.asarray(group)
    pred = (s >= threshold).astype(int)
    fnr = {}
    for lvl in np.unique(g):
        m = (g == lvl) & (y == 1)
        fnr[lvl] = float(np.mean(pred[m] == 0)) if m.any() else float("nan")
    vals = np.array([v for v in fnr.values() if np.isfinite(v)])
    return {"fnr": fnr, "gap": float(vals.max() - vals.min()) if vals.size else float("nan")}


def simulate_fairness_gap(n: int, prevalence_by_group: dict, fnr_by_group: dict, fpr_by_group: dict,
                          classifier_auc: float = 0.9, threshold_quantile: float = 0.8, n_rep: int = 200,
                          severity_dependence: float = 2.0, rng: np.random.Generator | None = None) -> dict:
    """Mechanism 1: measured FNR gap of a *fair* fixed classifier evaluated on differentially noisy labels.

    Scores ~ N(d', 1) for positives and N(0, 1) for negatives, identical across groups (true-label gap ≈ 0).
    The latent score doubles as severity, so non-mention noise concentrates on subtle positives.
    """
    from scipy.stats import norm

    rng = np.random.default_rng() if rng is None else rng
    groups = list(prevalence_by_group)
    dprime = np.sqrt(2) * norm.ppf(classifier_auc)
    gaps_noisy, gaps_true, aucs_noisy, fnr_noisy = [], [], [], {g: [] for g in groups}
    for _ in range(n_rep):
        g = rng.choice(groups, n)
        prev = np.array([prevalence_by_group[x] for x in g])
        y = (rng.random(n) < prev).astype(int)
        s = rng.standard_normal(n) + dprime * y
        thr = np.quantile(s, threshold_quantile)
        y_noisy = inject_noise(y, g, fnr_by_group, fpr_by_group, rng, severity=s, severity_dependence=severity_dependence)
        gaps_true.append(fnr_gap(s, y, g, thr)["gap"])
        res = fnr_gap(s, y_noisy, g, thr)
        gaps_noisy.append(res["gap"])
        for k, v in res["fnr"].items():
            fnr_noisy[k].append(v)
        aucs_noisy.append(roc_auc_score(y_noisy, s) if 0 < y_noisy.mean() < 1 else np.nan)
    return {"gap_true_labels": float(np.mean(gaps_true)), "gap_noisy_labels": float(np.mean(gaps_noisy)),
            "gap_noisy_ci": tuple(float(v) for v in np.percentile(gaps_noisy, [2.5, 97.5])),
            "fnr_noisy_by_group": {k: float(np.nanmean(v)) for k, v in fnr_noisy.items()},
            "auc_true": float(classifier_auc), "auc_noisy": float(np.nanmean(aucs_noisy))}


def simulate_training_under_noise(n_train: int, n_test: int, prevalence: float, fnr_by_group: dict,
                                  fpr_by_group: dict, signal: float = 2.0, n_noise_features: int = 5,
                                  severity_dependence: float = 2.0, threshold_quantile: float = 0.8,
                                  n_rep: int = 20, rng: np.random.Generator | None = None) -> dict:
    """Mechanism 2: a logistic model trained on differentially noisy labels, evaluated on expert labels.

    Features = [severity-linked signal, group one-hot, noise features]; the group indicator is available
    to the model (as image models implicitly encode it). The returned FNRs are computed against the
    *true* labels on a held-out test set with the threshold fixed on the training set.
    """
    rng = np.random.default_rng() if rng is None else rng
    groups = sorted(fnr_by_group)
    fnr_true_by_group = {g: [] for g in groups}
    gaps = []
    for _ in range(n_rep):
        def make(n: int):
            g = rng.choice(groups, n)
            y = (rng.random(n) < prevalence).astype(int)
            sev = rng.standard_normal(n) + signal * y
            X = np.column_stack([sev, np.stack([(g == k).astype(float) for k in groups], 1),
                                 rng.standard_normal((n, n_noise_features))])
            return X, y, g, sev

        Xtr, ytr, gtr, str_ = make(n_train)
        ytr_noisy = inject_noise(ytr, gtr, fnr_by_group, fpr_by_group, rng, severity=str_,
                                 severity_dependence=severity_dependence)
        model = LogisticRegression(max_iter=1000).fit(Xtr, ytr_noisy)
        thr = np.quantile(model.predict_proba(Xtr)[:, 1], threshold_quantile)
        Xte, yte, gte, _ = make(n_test)
        p = model.predict_proba(Xte)[:, 1]
        res = fnr_gap(p, yte, gte, thr)
        for k, v in res["fnr"].items():
            fnr_true_by_group[k].append(v)
        gaps.append(res["gap"])
    return {"fnr_true_labels_by_group": {k: float(np.mean(v)) for k, v in fnr_true_by_group.items()},
            "gap_true_labels": float(np.mean(gaps)),
            "gap_ci": tuple(float(v) for v in np.percentile(gaps, [2.5, 97.5]))}


def rank_agreement(auc_report: dict, auc_expert: dict) -> float:
    """Kendall tau between model rankings under report-derived vs expert labels."""
    from scipy.stats import kendalltau

    models = sorted(set(auc_report) & set(auc_expert))
    a = [auc_report[m] for m in models]
    b = [auc_expert[m] for m in models]
    return float(kendalltau(a, b).statistic) if len(models) >= 2 else float("nan")
