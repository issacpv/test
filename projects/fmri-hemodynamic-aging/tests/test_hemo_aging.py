"""Synthetic-data tests for hemo_aging (run with PYTHONPATH=src)."""
import numpy as np
import pytest

from hemo_aging import hrf_estimation as hrf
from hemo_aging import lag_mapping as lm
from hemo_aging import normative, stats


def _slfo(n: int, tr: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) * tr
    return (np.sin(2 * np.pi * 0.03 * t) + 0.6 * np.sin(2 * np.pi * 0.07 * t + 1.0)
            + 0.3 * rng.standard_normal(n).cumsum() / np.sqrt(n))


def test_lag_map_recovers_known_delays():
    rng = np.random.default_rng(0)
    tr, n = 0.8, 600
    base = _slfo(n + 100, tr, rng)
    true_lags = np.array([-2.0, 0.0, 1.5, 3.0, 4.5, 6.0])
    data = np.stack([lm.shift_series(base, l, tr)[50:50 + n] for l in true_lags])
    data += 0.3 * rng.standard_normal(data.shape)
    res = lm.estimate_lag_map(data, tr, max_lag=8, n_passes=3)
    rel = res.lag - res.lag[1]           # lags are relative to the reference; compare differences
    assert np.allclose(rel, true_lags - true_lags[1], atol=0.6)
    assert np.all(res.corr > 0.5)
    summ = lm.lag_summaries(res.lag, res.corr)
    assert summ["fit_fraction"] == 1.0 and summ["av_gradient"] > 3.0


def test_regress_out_slfo_reduces_shared_variance():
    rng = np.random.default_rng(1)
    tr, n = 0.8, 400
    base = _slfo(n, tr, rng)
    data = np.stack([lm.shift_series(base, l, tr) for l in (0.0, 2.0, 4.0)]) + 0.2 * rng.standard_normal((3, n))
    res = lm.estimate_lag_map(data, tr, max_lag=6, do_filter=False)
    cleaned = lm.regress_out_slfo(data, res, tr)
    before = np.abs(np.corrcoef(data)[np.triu_indices(3, 1)]).mean()
    after = np.abs(np.corrcoef(cleaned)[np.triu_indices(3, 1)]).mean()
    assert after < before


def test_hrf_time_to_peak_recovered():
    rng = np.random.default_rng(2)
    tr = 1.0
    true = hrf.canonical_hrf(tr, peak_delay=7.0)
    bold, _ = hrf.simulate_bold(2400, tr, true, event_rate=0.05, noise_sd=0.25, rng=rng)
    est, params = hrf.estimate_hrf(bold, tr, hrf_length=24, max_onset_delay=8, threshold=1.0)
    assert abs(params.time_to_peak - 6.0) <= 2.0   # canonical peak_delay=7 yields a mode near 6 s
    assert 2.0 < params.fwhm < 8.0
    assert params.r2 > 0.1


def test_hrf_parameters_of_canonical():
    tr = 0.5
    h = hrf.canonical_hrf(tr)
    p = hrf.hrf_parameters(h, tr)
    assert p.height == pytest.approx(1.0)
    assert 4.0 <= p.time_to_peak <= 6.0
    assert 3.0 <= p.fwhm <= 7.0
    assert p.undershoot < 0


def test_wiener_deconvolution_recovers_events():
    rng = np.random.default_rng(3)
    tr = 1.0
    h = hrf.canonical_hrf(tr)
    bold, events = hrf.simulate_bold(800, tr, h, event_rate=0.05, noise_sd=0.05, rng=rng)
    neural = hrf.wiener_deconvolve(bold, h, noise_ratio=0.01)
    r = np.corrcoef(neural, events - events.mean())[0, 1]
    r_raw = np.corrcoef(bold, events - events.mean())[0, 1]
    assert r > 0.5 and r > r_raw   # deconvolution sharpens the event train relative to raw BOLD


def test_normative_zscores_and_centiles():
    rng = np.random.default_rng(4)
    age = rng.uniform(36, 95, 1500)
    sd = 0.5 + 0.02 * (age - 36)
    y = 2.0 + 0.03 * (age - 36) + 0.0008 * (age - 36) ** 2 + sd * rng.standard_normal(age.size)
    model = normative.SplineNormativeModel(df=5).fit(age, y)
    z = model.zscore(age, y)
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.15
    c = model.centiles(np.array([40.0, 90.0]))
    assert c[0.95][1] - c[0.05][1] > c[0.95][0] - c[0.05][0]   # variance grows with age
    mu, _ = model.predict(np.array([40.0, 90.0]))
    assert mu[1] > mu[0]


def test_hemodynamic_age_delta_is_unbiased():
    rng = np.random.default_rng(5)
    n, p = 400, 30
    age = rng.uniform(40, 90, n)
    X = np.outer(age, rng.standard_normal(p)) / 50 + rng.standard_normal((n, p))
    out = normative.hemodynamic_age_delta(X, age, groups=np.arange(n), n_splits=5)
    assert out["r"] > 0.5
    assert abs(out["delta_age_corr"]) < 0.25


def test_icc_partial_corr_mediation():
    rng = np.random.default_rng(6)
    subj = rng.standard_normal(60)
    sessions = np.stack([subj + 0.3 * rng.standard_normal(60) for _ in range(2)], axis=1)
    assert 0.7 < stats.icc_2_1(sessions) < 1.0
    x = rng.standard_normal(300)
    cov = rng.standard_normal(300)
    m = 0.6 * x + 0.5 * rng.standard_normal(300)
    y = 0.5 * m + 0.2 * x + cov + 0.5 * rng.standard_normal(300)
    r, pval = stats.partial_corr(x, y, cov)
    assert r > 0.3 and pval < 1e-3
    med = stats.mediation_bootstrap(x, m, y, covars=cov, n_boot=200, rng=rng)
    assert med["indirect"] > 0.15 and med["indirect_ci_low"] > 0
    assert stats.attenuation_ratio(np.array([1.0, -1.0]), np.array([0.5, -0.5])) == pytest.approx(0.5)
