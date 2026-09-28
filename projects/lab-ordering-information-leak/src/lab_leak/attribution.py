"""Value-vs-ordering attribution, ordering-shift stress tests and order-agnostic training.

Core quantities
---------------
* ``two_player_shapley`` - exact Shapley split of a model's held-out log-likelihood
  gain (over the base rate) between the *value* channel and the *mask* channel.
  With two players the Shapley value needs only four fits (empty, V, M, V+M).
* ``thinning_stress_curve`` - AUROC / log-loss of fitted models when the test
  cohort's orders are thinned to emulate a lower-intensity ordering policy.
* ``fit_mask_dropout`` - order-agnostic training by *mask dropout*: each training
  stay is replicated with randomly deleted observations so the model cannot
  rely on any particular ordering density.
* ``intensity_weights`` - importance weights that equalise the ordering-intensity
  distribution between source and target (density-ratio via 1-D logistic fit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "lr", seed: int = 0):
    """``'lr'``: median-impute + scale + L2 logistic; ``'gbt'``: HistGradientBoosting (native NaN handling)."""
    if kind == "lr":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=2000)),
        ])
    if kind == "gbt":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    raise ValueError(kind)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def mean_loglik(y: np.ndarray, p: np.ndarray) -> float:
    return -log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def base_rate_loglik(y_train: np.ndarray, y_test: np.ndarray) -> float:
    p0 = float(np.clip(np.mean(y_train), 1e-6, 1 - 1e-6))
    return mean_loglik(y_test, np.full(len(y_test), p0))


@dataclass
class ChannelAttribution:
    """Held-out log-likelihood gains and their Shapley split (nats per stay)."""

    gain_values_only: float
    gain_masks_only: float
    gain_both: float
    shapley_values: float
    shapley_masks: float
    auroc_values_only: float
    auroc_masks_only: float
    auroc_both: float

    @property
    def mask_share(self) -> float:
        """Fraction of the joint model's gain attributable to the ordering channel.

        Returns 0 when the joint model does not beat the base rate (nothing to attribute).
        """
        return float(self.shapley_masks / self.gain_both) if self.gain_both > 0 else 0.0


def two_player_shapley(Xv_tr: pd.DataFrame, Xm_tr: pd.DataFrame, y_tr: np.ndarray,
                       Xv_te: pd.DataFrame, Xm_te: pd.DataFrame, y_te: np.ndarray,
                       kind: str = "lr", seed: int = 0) -> ChannelAttribution:
    """Exact two-player Shapley attribution of held-out log-likelihood gain.

    phi_M = 1/2 [ (LL(V+M) - LL(V)) + (LL(M) - LL(0)) ],  phi_V = gain_both - phi_M.
    """
    ll0 = base_rate_loglik(y_tr, y_te)
    fits = {}
    for name, (A_tr, A_te) in {"V": (Xv_tr, Xv_te), "M": (Xm_tr, Xm_te),
                               "VM": (pd.concat([Xv_tr, Xm_tr], axis=1), pd.concat([Xv_te, Xm_te], axis=1))}.items():
        m = make_model(kind, seed).fit(A_tr, y_tr)
        p = _predict(m, A_te)
        fits[name] = (mean_loglik(y_te, p) - ll0, roc_auc_score(y_te, p))
    gV, gM, gVM = fits["V"][0], fits["M"][0], fits["VM"][0]
    phi_M = 0.5 * ((gVM - gV) + gM)
    phi_V = gVM - phi_M
    return ChannelAttribution(gV, gM, gVM, phi_V, phi_M, fits["V"][1], fits["M"][1], fits["VM"][1])


def thinning_stress_curve(models: dict[str, object], feature_fn, long_te: pd.DataFrame, stays_te: pd.DataFrame,
                          keep_fracs: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25), n_rep: int = 3,
                          seed: int = 0, subset: str = "all", cfg=None) -> pd.DataFrame:
    """Evaluate each fitted model on progressively thinned test orders.

    ``models`` maps a name to ``(model, channel)`` with channel in {"V", "M", "VM"}.
    ``feature_fn(long, stays) -> (values, masks)``. Returns long-format results.
    """
    from .ordering import thin_orders  # local import to avoid a cycle at import time

    rng = np.random.default_rng(seed)
    y = stays_te["y"].to_numpy()
    rows = []
    for kf in keep_fracs:
        for r in range(n_rep):
            thinned = thin_orders(long_te, kf, rng, stays=stays_te, cfg=cfg, subset=subset)
            V, M = feature_fn(thinned, stays_te)
            for name, (model, ch) in models.items():
                X = {"V": V, "M": M, "VM": pd.concat([V, M], axis=1)}[ch]
                X = X.reindex(columns=getattr(model, "feature_names_in_", X.columns), fill_value=np.nan)
                p = _predict(model, X)
                rows.append({"model": name, "keep_frac": kf, "rep": r,
                             "auroc": roc_auc_score(y, p), "loglik": mean_loglik(y, p)})
    return pd.DataFrame(rows)


def degradation_area(curve: pd.DataFrame, metric: str = "auroc") -> pd.Series:
    """Area between the full-data metric and the thinned metric, per model (larger = more fragile)."""
    g = curve.groupby(["model", "keep_frac"])[metric].mean().unstack("keep_frac").sort_index(axis=1)
    full = g[1.0] if 1.0 in g.columns else g.iloc[:, -1]
    x = g.columns.to_numpy(dtype=float)
    loss = (full.to_numpy()[:, None] - g.to_numpy())
    return pd.Series(np.trapezoid(loss, x, axis=1) if hasattr(np, "trapezoid") else np.trapz(loss, x, axis=1), index=g.index, name="degradation_area")


def mask_dropout(values: pd.DataFrame, masks: pd.DataFrame, p: float, rng: np.random.Generator,
                 item_ids: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delete each observed lab (per stay, per item) with probability ``p``.

    Deleting an item sets its value features to NaN and its mask features to the
    'never ordered' state, and decrements the stay totals accordingly.
    """
    V, M = values.copy(), masks.copy()
    if item_ids is None:
        item_ids = sorted({c.rsplit("_", 1)[0] for c in V.columns})
    window = float(M.filter(like="_hrs_since").max().max()) if M.filter(like="_hrs_since").shape[1] else 0.0
    for it in item_ids:
        obs = M[f"{it}_any"].to_numpy() > 0
        drop = obs & (rng.random(len(M)) < p)
        if not drop.any():
            continue
        for c in [c for c in V.columns if c.startswith(f"{it}_")]:
            V.loc[drop, c] = np.nan
        M.loc[drop, "n_total"] = M.loc[drop, "n_total"] - M.loc[drop, f"{it}_n"]
        M.loc[drop, "n_stat_total"] = M.loc[drop, "n_stat_total"] - M.loc[drop, f"{it}_n_stat"]
        M.loc[drop, "n_off_total"] = M.loc[drop, "n_off_total"] - M.loc[drop, f"{it}_n_off"]
        M.loc[drop, "n_distinct"] = M.loc[drop, "n_distinct"] - 1
        M.loc[drop, [f"{it}_n", f"{it}_any", f"{it}_n_stat", f"{it}_n_off"]] = 0.0
        M.loc[drop, f"{it}_hrs_since"] = window
    return V, M


def fit_mask_dropout(values: pd.DataFrame, masks: pd.DataFrame, y: np.ndarray, p: float = 0.5,
                     n_aug: int = 3, kind: str = "lr", seed: int = 0, channel: str = "VM"):
    """Order-agnostic model: fit on the original plus ``n_aug`` mask-dropout replicas."""
    rng = np.random.default_rng(seed)
    Vs, Ms, ys = [values], [masks], [y]
    for _ in range(n_aug):
        V, M = mask_dropout(values, masks, p, rng)
        Vs.append(V)
        Ms.append(M)
        ys.append(y)
    V_all, M_all = pd.concat(Vs, ignore_index=True), pd.concat(Ms, ignore_index=True)
    X = {"V": V_all, "M": M_all, "VM": pd.concat([V_all, M_all], axis=1)}[channel]
    return make_model(kind, seed).fit(X, np.concatenate(ys))


def fit_order_dropout(long: pd.DataFrame, stays: pd.DataFrame, y: np.ndarray, feature_fn,
                      keep_fracs: tuple[float, ...] = (0.6, 0.35, 0.2), kind: str = "lr", seed: int = 0,
                      channel: str = "VM", subset: str = "all", cfg=None):
    """Order-agnostic model by *order-level dropout*: replicate the training cohort with its
    measurements thinned to several ordering intensities before feature building, then fit once.

    This matches the shape of a real ordering-policy shift (fewer draws per stay, same
    physiology) and is the variant that stays flattest under ``thinning_stress_curve``.
    ``feature_fn(long, stays) -> (values, masks)``.
    """
    from .ordering import thin_orders

    rng = np.random.default_rng(seed)
    V0, M0 = feature_fn(long, stays)
    Vs, Ms, ys = [V0], [M0], [np.asarray(y)]
    for kf in keep_fracs:
        thinned = thin_orders(long, kf, rng, stays=stays, cfg=cfg, subset=subset)
        V, M = feature_fn(thinned, stays)
        Vs.append(V)
        Ms.append(M)
        ys.append(np.asarray(y))
    V_all, M_all = pd.concat(Vs, ignore_index=True), pd.concat(Ms, ignore_index=True)
    X = {"V": V_all, "M": M_all, "VM": pd.concat([V_all, M_all], axis=1)}[channel]
    return make_model(kind, seed).fit(X, np.concatenate(ys))


def intensity_weights(n_source: np.ndarray, n_target: np.ndarray) -> np.ndarray:
    """Importance weights w(x) = P(target|n)/P(source|n) * n_S/n_T from a 1-D logistic domain classifier
    on log(1 + orders per stay). Weights are normalised to mean 1 over the source."""
    x = np.log1p(np.concatenate([n_source, n_target]).astype(float))[:, None]
    d = np.r_[np.zeros(len(n_source)), np.ones(len(n_target))]
    clf = LogisticRegression(C=1.0).fit(x, d)
    pt = clf.predict_proba(np.log1p(np.asarray(n_source, dtype=float))[:, None])[:, 1]
    w = pt / np.clip(1 - pt, 1e-6, None) * (len(n_source) / len(n_target))
    return w / w.mean()
