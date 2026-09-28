"""Tests for the AST leakage detector on synthetic code snippets and a tiny notebook."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from leakscan import findings_to_frame, prevalence_table, scan_file, scan_path, scan_source, summarise_repo, to_markdown
from leakscan.corpus import extract_repo_links
from leakscan.report import wilson_interval


def rules(src: str, **kw) -> set[str]:
    return scan_source(textwrap.dedent(src), **kw).rules()


def sev(src: str, rule: str, **kw) -> str:
    res = scan_source(textwrap.dedent(src), **kw)
    return next(f.severity for f in res.findings if f.rule == rule)


# --------------------------------------------------------------------------- #
# split-without-groups / window leakage
# --------------------------------------------------------------------------- #
def test_random_split_on_subject_data_flagged_high():
    src = """
    import numpy as np
    from sklearn.model_selection import train_test_split
    X, y, subject_id = load_chbmit_windows("data/chbmit")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    """
    r = rules(src)
    assert "split-without-groups" in r
    assert "window-level-random-split" in r
    assert sev(src, "split-without-groups") == "high"


def test_group_split_not_flagged():
    src = """
    from sklearn.model_selection import GroupKFold
    X, y, groups = load_patients()
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        pass
    """
    r = rules(src)
    assert "split-without-groups" not in r
    assert "group-aware-split-present" in r


def test_cross_val_score_with_groups_kw_is_group_aware():
    src = """
    from sklearn.model_selection import cross_val_score, GroupKFold
    scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=patient_id)
    """
    r = rules(src)
    assert "split-without-groups" not in r


def test_random_split_without_subject_evidence_is_low():
    src = """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    """
    assert sev(src, "split-without-groups") == "low"
    assert sev(src, "split-without-groups", subject_level=True) == "high"


def test_kfold_split_method_flagged():
    src = """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for tr, te in skf.split(X_epochs, y):
        model.fit(X_epochs[tr], y[tr])
    """
    r = rules(src, subject_level=True)
    assert "split-without-groups" in r
    assert "window-level-random-split" in r


def test_mixed_splitting_is_low_note():
    src = """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1)
    a, b = next(gss.split(X, y, groups=subjects))
    X_tr, X_te = train_test_split(X_small)
    """
    res = scan_source(textwrap.dedent(src))
    assert "mixed-splitting" in res.rules()
    assert "split-without-groups" not in res.rules()


# --------------------------------------------------------------------------- #
# preprocessing / selection / resampling before split
# --------------------------------------------------------------------------- #
def test_scaler_fit_before_split_flagged():
    src = """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    """
    assert "preprocess-before-split" in rules(src)


def test_scaler_fit_via_output_variable_flagged():
    src = """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    X_scaled = MinMaxScaler().fit_transform(raw)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    """
    assert "preprocess-before-split" in rules(src)


def test_scaler_fit_on_train_only_not_flagged():
    src = """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    """
    r = rules(src)
    assert "preprocess-before-split" not in r
    assert "test-set-in-training" not in r


def test_feature_selection_before_split_is_high():
    src = """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold
    sel = SelectKBest(f_classif, k=20)
    X_sel = sel.fit_transform(X, y)
    cv = StratifiedKFold(5)
    for tr, te in cv.split(X_sel, y):
        pass
    """
    assert sev(src, "feature-selection-before-split") == "high"


def test_smote_before_split_flagged():
    src = """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    X_res, y_res = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
    """
    assert "resample-before-split" in rules(src)


def test_pipeline_in_cv_is_positive_evidence():
    src = """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=subject)
    """
    r = rules(src)
    assert "pipeline-present" in r
    assert "preprocess-before-split" not in r


# --------------------------------------------------------------------------- #
# test-set misuse and no hold-out
# --------------------------------------------------------------------------- #
def test_keras_validation_data_is_test_set():
    src = """
    model = Sequential()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stop])
    """
    assert "test-set-in-training" in rules(src)


def test_gridsearch_fit_on_test_flagged():
    src = """
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(clf, params, cv=5)
    gs.fit(X_test, y_test)
    """
    assert "test-set-in-training" in rules(src)


def test_scaler_fitted_on_test_flagged():
    src = """
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_test = StandardScaler().fit_transform(X_test)
    """
    assert "test-set-in-training" in rules(src)


def test_no_holdout_evaluation():
    src = """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    clf = RandomForestClassifier()
    clf.fit(X, y)
    pred = clf.predict(X)
    print(accuracy_score(y, pred))
    """
    assert "no-holdout-evaluation" in rules(src)


def test_clean_pipeline_has_no_leakage_rules():
    src = """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    for tr, te in GroupKFold(5).split(X, y, groups=patient_id):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression().fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))
    """
    res = scan_source(textwrap.dedent(src))
    leak = {f.rule for f in res.findings if f.severity in ("high", "medium")}
    assert leak == set()


def test_fit_and_predict_in_different_methods_not_flagged():
    src = """
    class Wrapper:
        def fit(self, X, y):
            self.est.fit(X, y)
            return self
        def predict(self, X):
            return self.est.predict_proba(X)[:, 1]
    """
    assert "no-holdout-evaluation" not in rules(src)


def test_scaler_in_helper_function_and_split_elsewhere_not_ordered():
    src = """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    def scale(X):
        return StandardScaler().fit_transform(X)
    def split(X, y):
        return train_test_split(X, y)
    """
    assert "preprocess-before-split" not in rules(src)


def test_syntax_error_reported_not_raised():
    res = scan_source("def broken(:\n  pass\n")
    assert res.parse_error is not None
    assert "parse-error" in res.rules()


# --------------------------------------------------------------------------- #
# notebooks, files and reports
# --------------------------------------------------------------------------- #
def _write_notebook(path: Path, cells: list[str]) -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": c} for c in cells]}
    path.write_text(json.dumps(nb))


def test_notebook_cross_cell_ordering(tmp_path):
    nb = tmp_path / "analysis.ipynb"
    _write_notebook(
        nb,
        [
            "%matplotlib inline\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n",
            "X, y, patient = load_ptbxl('data/ptbxl_database.csv')\nX = StandardScaler().fit_transform(X)\n",
            "from sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)\n",
        ],
    )
    res = scan_file(nb)
    r = res.rules()
    assert "preprocess-before-split" in r and "split-without-groups" in r
    pre = next(f for f in res.findings if f.rule == "preprocess-before-split")
    assert pre.cell == 1  # second cell (0-based over all cells)


def test_scan_path_and_report(tmp_path):
    (tmp_path / "a.py").write_text("from sklearn.model_selection import train_test_split\nX_tr, X_te = train_test_split(X_subject_windows)\n")
    (tmp_path / "b.py").write_text("print('hello')\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "c.py").write_text("train_test_split(X)\n")
    results = scan_path(tmp_path)
    assert {Path(r.file).name for r in results} == {"a.py", "b.py"}
    df = findings_to_frame(results, repo="owner/repo")
    assert (df["rule"] == "split-without-groups").any()
    repo_df = summarise_repo(df, min_severity="low")
    assert bool(repo_df.loc[0, "split-without-groups"])
    prev = prevalence_table(repo_df)
    assert set(prev.columns) >= {"rule", "k", "n", "prevalence", "ci_low", "ci_high"}
    md = to_markdown(results)
    assert "split-without-groups" in md


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # nan


def test_extract_repo_links():
    txt = "Code: https://github.com/lab/ecg-model.git and https://github.com/lab/ecg-model/tree/main, also github.com/other/thing"
    assert extract_repo_links(txt) == ["lab/ecg-model"]
