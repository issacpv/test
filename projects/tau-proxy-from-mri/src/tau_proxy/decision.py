"""Decision-curve analysis and tau-PET triage policy model.

Net benefit follows Vickers & Elkin (2006): NB(t) = TP/n − FP/n · t/(1−t).
The policy model asks: if everyone with predicted probability below a cut-off
chosen to keep sensitivity ≥ s is *not* sent to tau PET, how many scans are
avoided per 1,000 screened, how many T+ are missed, and what is the cost per
identified T+ participant? Comparator arms (e.g., plasma p-tau217 at a
published sensitivity/specificity) are entered as parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def net_benefit_curve(prob: np.ndarray, y: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Net benefit of the model, treat-all and treat-none across thresholds."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    prev = y.mean()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.70, 27)
    rows = []
    for t in thresholds:
        pred = prob >= t
        tp = np.sum(pred & (y == 1)) / n
        fp = np.sum(pred & (y == 0)) / n
        w = t / (1 - t)
        rows.append({"threshold": t, "nb_model": tp - fp * w, "nb_all": prev - (1 - prev) * w, "nb_none": 0.0})
    return pd.DataFrame(rows)


@dataclass
class TriageCosts:
    pet_cost: float = 3500.0          # per tau PET scan (USD, parameter)
    mri_marginal_cost: float = 0.0    # 0 if the MRI is already acquired
    blood_test_cost: float = 300.0    # comparator arm
    n_screened: int = 1000


def triage_policy(prob: np.ndarray, y: np.ndarray, sensitivity: float, costs: TriageCosts) -> dict[str, float]:
    """Scans avoided and cost per identified T+ at a fixed sensitivity target."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    pos = prob[y == 1]
    cut = np.quantile(pos, 1 - sensitivity, method="lower") if pos.size else np.nan
    sent = prob >= cut
    sens = np.sum(sent & (y == 1)) / max(1, np.sum(y == 1))
    spec = np.sum(~sent & (y == 0)) / max(1, np.sum(y == 0))
    frac_sent = sent.mean()
    n = costs.n_screened
    prev = y.mean()
    scans = frac_sent * n
    identified = sens * prev * n
    total_cost = scans * costs.pet_cost + n * costs.mri_marginal_cost
    return {
        "cutoff": float(cut),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "scans_per_1000": float(scans * 1000 / n),
        "scans_avoided_per_1000": float((1 - frac_sent) * 1000),
        "missed_Tpos_per_1000": float((1 - sens) * prev * 1000),
        "cost_per_identified_Tpos": float(total_cost / identified) if identified > 0 else float("nan"),
    }


def comparator_policy(sensitivity: float, specificity: float, prevalence: float, costs: TriageCosts) -> dict[str, float]:
    """Same policy metrics for a test summarized by published sensitivity/specificity (e.g. plasma p-tau217)."""
    n = costs.n_screened
    frac_sent = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    scans = frac_sent * n
    identified = sensitivity * prevalence * n
    total_cost = scans * costs.pet_cost + n * costs.blood_test_cost
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "scans_per_1000": float(scans * 1000 / n),
        "scans_avoided_per_1000": float((1 - frac_sent) * 1000),
        "missed_Tpos_per_1000": float((1 - sensitivity) * prevalence * 1000),
        "cost_per_identified_Tpos": float(total_cost / identified) if identified > 0 else float("nan"),
    }


def no_prescreen_policy(prevalence: float, costs: TriageCosts) -> dict[str, float]:
    n = costs.n_screened
    return {
        "sensitivity": 1.0, "specificity": 0.0, "scans_per_1000": 1000.0, "scans_avoided_per_1000": 0.0,
        "missed_Tpos_per_1000": 0.0,
        "cost_per_identified_Tpos": float(n * costs.pet_cost / (prevalence * n)) if prevalence > 0 else float("nan"),
    }


def policy_table(models: dict[str, np.ndarray], y: np.ndarray, sensitivities=(0.8, 0.9, 0.95),
                 costs: TriageCosts | None = None, comparators: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Policy metrics for each model × sensitivity target plus comparator and no-prescreen rows."""
    costs = costs or TriageCosts()
    prev = float(np.mean(y))
    rows = []
    for s in sensitivities:
        for name, p in models.items():
            rows.append({"arm": name, "target_sensitivity": s, **triage_policy(p, y, s, costs)})
    for name, (se, sp) in (comparators or {}).items():
        rows.append({"arm": name, "target_sensitivity": se, **comparator_policy(se, sp, prev, costs)})
    rows.append({"arm": "no_prescreen", "target_sensitivity": 1.0, **no_prescreen_policy(prev, costs)})
    return pd.DataFrame(rows)
