"""The evaluation harness: run shift cells, build leaderboards, regress loss on diagnostics.

A model is anything with ``fit(X, y, groups=None)`` and ``predict(X)`` returning
scores of the right shape (probabilities for binary/event windows, a (n, k)
probability matrix for multilabel, a value for regression).  The harness:

1. loads source ``train`` and ``test`` splits and each target's full data
   through the adapter;
2. fits the model on source train, evaluates on source test (in-domain) and
   on each target (transfer) with group bootstrap;
3. computes diagnostics (domain-classifier AUC, MMD) on ``X`` and, for binary
   tasks, the covariate/label/concept decomposition;
4. returns one long-format row per (cell, evaluation kind, metric).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .adapters import Adapter, DomainData, group_split
from .diagnostics import decompose_gap, domain_classifier_auc, mmd_rbf, proxy_a_distance
from .metrics import evaluate
from .spec import Benchmark, ShiftAxis, ShiftCell, TaskType, default_benchmark


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> "Model": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SklearnBinaryModel:
    """Wrap any sklearn classifier / regressor into the harness ``Model`` protocol."""

    def __init__(self, estimator, task_type: TaskType | str = TaskType.BINARY) -> None:
        self.est = estimator
        self.task_type = TaskType(getattr(task_type, "value", task_type))

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> "SklearnBinaryModel":
        self.est.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == TaskType.REGRESSION:
            return np.asarray(self.est.predict(X), float)
        if self.task_type == TaskType.MULTILABEL:
            proba = self.est.predict_proba(X)
            if isinstance(proba, list):  # OneVsRest / MultiOutput style
                return np.column_stack([p[:, 1] for p in proba])
            return np.asarray(proba)
        return np.asarray(self.est.predict_proba(X)[:, 1], float)


def default_model_factory(task_type: TaskType) -> Model:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if task_type == TaskType.REGRESSION:
        return SklearnBinaryModel(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), task_type)
    if task_type == TaskType.MULTILABEL:
        return SklearnBinaryModel(MultiOutputClassifier(LogisticRegression(max_iter=500)), task_type)
    return SklearnBinaryModel(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), task_type)


def _rows(cell: ShiftCell, task_modality: str, kind: str, domain: str, metrics: Dict[str, float], extra: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    base = {
        "cell": cell.id,
        "task": cell.task,
        "modality": task_modality,
        "source": "+".join(cell.source),
        "target": "+".join(cell.target),
        "axes": ";".join(sorted(a.value for a in cell.axes)),
        "kind": kind,
        "domain": domain,
    }
    if extra:
        base.update(extra)
    return [dict(base, metric=k, value=v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))]


def _concat(datas: Sequence[DomainData]) -> DomainData:
    if len(datas) == 1:
        return datas[0]
    X = np.vstack([d.X for d in datas])
    y = np.concatenate([d.y for d in datas])
    groups = np.concatenate([np.char.add(f"{d.domain}:", d.groups.astype(str)) for d in datas])
    meta = pd.concat([d.meta for d in datas], ignore_index=True)
    return DomainData(X, y, groups, meta, "+".join(d.domain for d in datas), datas[0].task)


def run_cell(cell: ShiftCell, benchmark: Benchmark, adapter: Adapter, model_factory: Callable[[TaskType], Model] = default_model_factory, n_boot: int = 100, seed: int = 0, diagnostics: bool = True, model_name: str = "baseline") -> pd.DataFrame:
    """Evaluate one shift cell; returns long-format rows."""
    task = benchmark.tasks[cell.task]
    t0 = time.time()
    src_train = _concat([adapter.load(cell.task, d, "train") for d in cell.source])
    src_test = _concat([adapter.load(cell.task, d, "test") for d in cell.source])
    model = model_factory(task.task_type).fit(src_train.X, src_train.y, src_train.groups)
    s_src = model.predict(src_test.X)
    m_src = evaluate(task.task_type, src_test.y, s_src, src_test.groups, src_test.meta, task.subgroup_columns, n_boot, seed)
    extra = {"model": model_name, "contaminated": model_name in cell.contaminated_models}
    rows = _rows(cell, task.modality, "in_domain", src_test.domain, m_src, extra)
    for d in cell.target:
        tgt = adapter.load(cell.task, d, "all")
        s_tgt = model.predict(tgt.X)
        m_tgt = evaluate(task.task_type, tgt.y, s_tgt, tgt.groups, tgt.meta, task.subgroup_columns, n_boot, seed)
        rows += _rows(cell, task.modality, "transfer", d, m_tgt, extra)
        gap = {f"gap_{k}": m_tgt[k] - m_src[k] for k in m_tgt if k in m_src and isinstance(m_tgt[k], float) and k not in ("n",)}
        rows += _rows(cell, task.modality, "gap", d, gap, extra)
        if diagnostics:
            diag = {"domain_auc": domain_classifier_auc(src_test.X, tgt.X, seed=seed, groups_s=src_test.groups, groups_t=tgt.groups)}
            diag["proxy_a_distance"] = proxy_a_distance(diag["domain_auc"])
            mmd, p = mmd_rbf(src_test.X, tgt.X, n_perm=50, seed=seed)
            diag["mmd"], diag["mmd_p"] = mmd, p
            if task.task_type in (TaskType.BINARY, TaskType.EVENT_DETECTION):
                diag.update({f"decomp_{k}": v for k, v in decompose_gap(src_test.y, s_src, src_test.X, tgt.y, s_tgt, tgt.X, seed=seed).items()})
            rows += _rows(cell, task.modality, "diagnostic", d, diag, extra)
    df = pd.DataFrame(rows)
    df["seconds"] = time.time() - t0
    return df


def run_benchmark(benchmark: Optional[Benchmark], adapter: Adapter, model_factory: Callable[[TaskType], Model] = default_model_factory, cells: Optional[Iterable[ShiftCell]] = None, n_boot: int = 100, seed: int = 0, diagnostics: bool = True, model_name: str = "baseline", verbose: bool = False) -> pd.DataFrame:
    """Run many cells and concatenate their rows."""
    benchmark = benchmark or default_benchmark()
    frames = []
    for c in cells if cells is not None else benchmark.cells:
        if verbose:
            print(f"[bioshift] {c.id}", flush=True)
        frames.append(run_cell(c, benchmark, adapter, model_factory, n_boot, seed, diagnostics, model_name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def leaderboard(results: pd.DataFrame, primary_only: bool = True) -> pd.DataFrame:
    """Wide table: one row per (model, cell, target) with in-domain, transfer, gap, calibration and diagnostics."""
    if results.empty:
        return results
    keep = ["primary", "cal_intercept", "cal_slope", "ece", "gap_sex", "gap_age_band", "auroc", "macro_auroc", "mae", "event_f1"]
    r = results[results["metric"].isin(keep + [f"gap_{k}" for k in keep] + ["domain_auc", "mmd", "decomp_covariate", "decomp_label", "decomp_concept"])]
    r = r.assign(col=r["kind"].str.cat(r["metric"], sep=":"))
    wide = r.pivot_table(index=["model", "modality", "task", "cell", "source", "target", "axes", "contaminated"], columns="col", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    if primary_only:
        cols = [c for c in wide.columns if not isinstance(c, str) or ":" not in c or c.endswith((":primary", ":cal_intercept", ":cal_slope", ":ece", ":gap_sex", ":gap_age_band", ":domain_auc", ":mmd", ":decomp_covariate", ":decomp_label", ":decomp_concept"))]
        wide = wide[cols]
    return wide.sort_values(["modality", "task", "cell"]).reset_index(drop=True)


def shift_loss_regression(results: pd.DataFrame, diagnostic: str = "domain_auc", loss: str = "gap_primary") -> pd.DataFrame:
    """Spearman correlation between a diagnostic and the primary-metric gap, pooled and per modality."""
    d = results[(results["kind"] == "diagnostic") & (results["metric"] == diagnostic)][["model", "cell", "target", "modality", "value"]].rename(columns={"value": "diag"})
    g = results[(results["kind"] == "gap") & (results["metric"] == loss)][["model", "cell", "target", "value"]].rename(columns={"value": "loss"})
    m = d.merge(g, on=["model", "cell", "target"])
    rows = []
    for name, sub in [("pooled", m)] + list(m.groupby("modality")):
        if len(sub) >= 3 and sub["diag"].std() > 0 and sub["loss"].std() > 0:
            rho, p = stats.spearmanr(sub["diag"], sub["loss"])
        else:
            rho, p = float("nan"), float("nan")
        rows.append({"stratum": name, "n_cells": len(sub), "spearman_rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


def axis_summary(results: pd.DataFrame, benchmark: Optional[Benchmark] = None) -> pd.DataFrame:
    """Mean primary-metric gap per shift axis (cells tagged with the axis vs not), per modality."""
    g = results[(results["kind"] == "gap") & (results["metric"] == "gap_primary")]
    rows = []
    for axis in ShiftAxis:
        has = g[g["axes"].str.contains(axis.value)]
        not_has = g[~g["axes"].str.contains(axis.value)]
        for mod in sorted(g["modality"].unique()):
            a, b = has[has["modality"] == mod]["value"], not_has[not_has["modality"] == mod]["value"]
            rows.append({"modality": mod, "axis": axis.value, "n_with": len(a), "n_without": len(b), "mean_gap_with": float(a.mean()) if len(a) else float("nan"), "mean_gap_without": float(b.mean()) if len(b) else float("nan")})
    return pd.DataFrame(rows)


def negative_control(task_id: str, domain_id: str, benchmark: Benchmark, adapter: Adapter, model_factory: Callable[[TaskType], Model] = default_model_factory, seed: int = 0, n_boot: int = 50) -> Dict[str, float]:
    """Split one domain into two group-disjoint halves and run the 'transfer' between them (gap should be ~0)."""
    task = benchmark.tasks[task_id]
    data = adapter.load(task_id, domain_id, "all")
    half = group_split(data.groups, 0.5, seed=seed)
    a, b = data.subset(~half, "half_a"), data.subset(half, "half_b")
    inner = group_split(a.groups, 0.3, seed=seed + 1)
    model = model_factory(task.task_type).fit(a.X[~inner], a.y[~inner], a.groups[~inner])
    m_in = evaluate(task.task_type, a.y[inner], model.predict(a.X[inner]), a.groups[inner], a.meta.iloc[np.flatnonzero(inner)].reset_index(drop=True), task.subgroup_columns, n_boot, seed)
    m_out = evaluate(task.task_type, b.y, model.predict(b.X), b.groups, b.meta, task.subgroup_columns, n_boot, seed)
    return {"in_domain": m_in["primary"], "other_half": m_out["primary"], "gap": m_out["primary"] - m_in["primary"], "domain_auc": domain_classifier_auc(a.X, b.X, seed=seed, groups_s=a.groups, groups_t=b.groups)}
