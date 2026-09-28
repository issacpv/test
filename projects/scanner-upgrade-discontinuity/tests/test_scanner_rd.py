"""Synthetic-data tests for scanner_rd (run with PYTHONPATH=src)."""

import numpy as np
import pandas as pd
import pytest

from scanner_rd import (
    ComBat,
    LongitudinalComBatLite,
    add_event_time,
    aging_equivalent_years,
    build_session_table,
    find_paired_sessions,
    find_transitions,
    local_linear_rd,
    parse_oasis_id,
    placebo_cutoffs,
    rd_anchored_correction,
    simulate_cohort,
)
from scanner_rd.harmonize import batch_effect_size


def test_parse_oasis_id():
    assert parse_oasis_id("OAS30001_MR_d0129") == ("OAS30001", 129)
    with pytest.raises(ValueError):
        parse_oasis_id("sub-01")


def test_transitions_and_event_time():
    mr = pd.DataFrame(
        {
            "MR ID": ["OAS30001_MR_d0000", "OAS30001_MR_d0400", "OAS30001_MR_d0800", "OAS30002_MR_d0010", "OAS30002_MR_d0020"],
            "Scanner": ["Trio", "Trio", "mMR", "Trio", "mMR"],
        }
    )
    sess = build_session_table(mr)
    tr = find_transitions(sess)
    assert len(tr) == 2
    t1 = tr[tr.subject == "OAS30001"].iloc[0]
    assert (t1.from_scanner, t1.to_scanner, t1.cutoff_day, t1.n_pre, t1.n_post) == ("Trio", "mMR", 800, 2, 1)
    ev = add_event_time(sess, tr, ("Trio", "mMR"), min_pre=2)
    assert set(ev.subject) == {"OAS30001"}
    assert ev.loc[ev.day == 800, "t_rel"].iloc[0] == 0.0
    assert ev.loc[ev.day == 0, "post"].iloc[0] == 0
    pairs = find_paired_sessions(sess, max_days=14)
    assert list(pairs.subject) == ["OAS30002"]


def test_rdit_recovers_jump():
    df = simulate_cohort(n_subjects=300, jump=80.0, noise_sd=30.0, seed=1)
    df = df.dropna(subset=["t_rel"])
    res = local_linear_rd(df, "y", bandwidth=2.5)
    assert abs(res.jump - 80.0) < 3 * res.jump_se + 5
    assert res.jump_se < 15
    assert res.n_subjects > 100
    # slope is negative (atrophy) and there is no slope change by construction
    assert res.slope_pre < 0
    assert abs(res.slope_change) < 3 * res.slope_change_se + 5


def test_rdit_null_and_placebo():
    df = simulate_cohort(n_subjects=300, jump=0.0, noise_sd=30.0, seed=2).dropna(subset=["t_rel"])
    res = local_linear_rd(df, "y", bandwidth=2.5)
    assert abs(res.jump) < 3 * res.jump_se + 5
    df2 = simulate_cohort(n_subjects=400, jump=80.0, sessions_per_subject=(6, 7, 8), noise_sd=30.0, seed=3)
    df2 = df2.dropna(subset=["t_rel"])
    pl = placebo_cutoffs(df2, "y", cutoffs=[-1.5, 1.5], bandwidth=1.5)
    ok = pl.dropna()
    assert len(ok) >= 1
    assert (ok.jump.abs() < 3 * ok.se + 8).all()


def test_aging_equivalent_years():
    assert aging_equivalent_years(80.0, -40.0) == pytest.approx(2.0)
    assert np.isnan(aging_equivalent_years(80.0, 0.0))


def test_combat_reduces_batch_effect():
    rng = np.random.default_rng(0)
    n, p = 200, 6
    batch = np.repeat(["A", "B"], n // 2)
    age = rng.uniform(50, 90, n)
    X = 3000 - 20 * age[:, None] + rng.normal(0, 100, (n, p))
    X[batch == "B"] += 150.0
    X[batch == "B"] *= 1.2
    before = np.abs(batch_effect_size(X, batch)).mean()
    Xh = ComBat().fit_transform(X, batch, covars=age)
    after = np.abs(batch_effect_size(Xh, batch)).mean()
    assert after < 0.25 * before
    # the biological covariate effect survives harmonization
    slope = np.polyfit(age, Xh[:, 0], 1)[0]
    assert -30 < slope < -10


def test_longitudinal_combat_and_rd_anchor():
    df = simulate_cohort(n_subjects=250, jump=80.0, scale=1.0, noise_sd=20.0, seed=4)
    X = df[["y"]].to_numpy()
    lc = LongitudinalComBatLite().fit(X, df.scanner.to_numpy(), df.subject.to_numpy(), covars=df.years.to_numpy())
    # gamma_B - gamma_A should be close to the true jump
    gA, gB = lc.gamma_[list(lc.batches_).index("A"), 0], lc.gamma_[list(lc.batches_).index("B"), 0]
    assert abs((gB - gA) - 80.0) < 15
    Xh = lc.transform(X, df.scanner.to_numpy(), subject=df.subject.to_numpy(), covars=df.years.to_numpy())
    tr = df.dropna(subset=["t_rel"]).copy()
    tr["y_h"] = Xh[tr.index, 0]
    res = local_linear_rd(tr, "y_h", bandwidth=2.5)
    assert abs(res.jump) < 3 * res.jump_se + 8
    # RD-anchored correction with the true jump removes the discontinuity
    tr["y_rd"] = rd_anchored_correction(tr.y.to_numpy(), tr.t_rel.to_numpy(), jump=80.0)
    res2 = local_linear_rd(tr, "y_rd", bandwidth=2.5)
    assert abs(res2.jump) < 3 * res2.jump_se + 8
