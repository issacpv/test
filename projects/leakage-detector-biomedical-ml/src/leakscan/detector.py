"""AST-based leakage-pattern detector for Python ML code (scripts and notebooks).

The detector is deliberately *intra-file and order-based*: it walks the AST in
source order, records "events" (splits, transformer fits, estimator fits,
evaluations) with the root variable names they touch, and then applies a small
set of rules that compare event order and name overlap.  This mirrors how a
careful human reviewer reads a pipeline ("was the scaler fitted before the
split? is there a groups= argument anywhere?"), is robust to notebooks, and
needs no execution of untrusted code.

It is a *screening* tool: every rule has documented false-positive modes (see
README "Risks"), and findings are meant to be verified by a human annotator in
the audit.  Severity is graded by the amount of subject-level evidence found in
the same file.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .notebooks import NotebookSource, notebook_to_source

# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #
RANDOM_SPLITTERS: Set[str] = {
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "random_split",  # torch.utils.data.random_split
}
GROUP_SPLITTERS: Set[str] = {
    "GroupKFold",
    "StratifiedGroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "PredefinedSplit",
    "TimeSeriesSplit",
}
CV_FUNCS: Set[str] = {
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
    "learning_curve",
    "validation_curve",
    "permutation_test_score",
}
SEARCH_CLASSES: Set[str] = {
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
}
TRANSFORMERS: Set[str] = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "Normalizer",
    "PowerTransformer",
    "QuantileTransformer",
    "PolynomialFeatures",
    "PCA",
    "KernelPCA",
    "IncrementalPCA",
    "TruncatedSVD",
    "FastICA",
    "NMF",
    "FactorAnalysis",
    "LinearDiscriminantAnalysis",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "LabelEncoder",
    "KBinsDiscretizer",
    "TfidfVectorizer",
    "CountVectorizer",
}
FEATURE_SELECTORS: Set[str] = {
    "SelectKBest",
    "SelectPercentile",
    "SelectFpr",
    "SelectFdr",
    "SelectFwe",
    "GenericUnivariateSelect",
    "RFE",
    "RFECV",
    "SelectFromModel",
    "SequentialFeatureSelector",
    "VarianceThreshold",
}
RESAMPLERS: Set[str] = {
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "SVMSMOTE",
    "KMeansSMOTE",
    "SMOTENC",
    "SMOTEN",
    "RandomOverSampler",
    "RandomUnderSampler",
    "SMOTEENN",
    "SMOTETomek",
    "TomekLinks",
    "NearMiss",
    "ClusterCentroids",
}
PIPELINES: Set[str] = {"Pipeline", "make_pipeline", "ColumnTransformer", "make_column_transformer"}
ESTIMATORS: Set[str] = {
    "LogisticRegression",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVC",
    "SVR",
    "LinearSVC",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "AdaBoostClassifier",
    "GaussianNB",
    "MLPClassifier",
    "MLPRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "Sequential",
    "Model",
    "TabNetClassifier",
    "MLPClassifier",
}
FIT_METHODS: Set[str] = {"fit", "fit_transform", "fit_resample", "fit_predict", "partial_fit"}
EVAL_METHODS: Set[str] = {"predict", "predict_proba", "score", "evaluate", "decision_function", "transform"}
METRIC_FUNCS: Set[str] = {
    "accuracy_score",
    "roc_auc_score",
    "f1_score",
    "precision_score",
    "recall_score",
    "balanced_accuracy_score",
    "average_precision_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "confusion_matrix",
    "classification_report",
}

# Names that hint at subject/patient-level structure, sliding windows, or the
# biomedical datasets we audit.  Applied to identifiers *and* string constants.
SUBJECT_TOKEN_RE = re.compile(
    r"subject|subj\b|subjs?\b|patient|participant|recording|record_?id|"
    r"\bpid\b|\bsid\b|chb\d{2}|chbmit|chb[-_ ]?mit|ptb[-_ ]?xl|ecg_id|mimic|eicu|hirid|"
    r"physionet|wfdb|pyedflib|\bmne\b|\.edf\b|\.nwb\b|\.nii\b|nsrr|\btuh\b|tusz|siena|"
    r"\bdeap\b|eegmmidb|sleep[-_ ]?edf|\bshhs\b|stay_id|hadm_id|icustay|\bbonn\b|"
    r"echonet|\bcamus\b|code[-_ ]?15|chapman|ningbo",
    re.I,
)
WINDOW_TOKEN_RE = re.compile(
    r"window|windows|windowed|sliding|epoch(?!s?\s*=)|epochs\b|segment|segments|segmented|"
    r"crop|crops|frame|frames|stride|overlap|sliding_window_view|chunk|chunks",
    re.I,
)
TEST_NAME_RE = re.compile(r"(^|_)(x|y|X|Y|data|df|set|idx|ids|loader|ds|labels?|features?)?_?tests?($|_)|^tests?_(x|y|X|Y|data|df|set|idx|ids|loader|ds|labels?|features?)($|_)")
TRAIN_NAME_RE = re.compile(r"(^|_)tr(ain|n)($|_)|(^|_)train_?(x|y|X|Y|data|df)($|_)", re.I)

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1, "info": 0}


# --------------------------------------------------------------------------- #
# data classes
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    """One detected pattern."""

    rule: str
    severity: str
    lineno: int
    message: str
    file: str = ""
    cell: Optional[int] = None
    cell_line: Optional[int] = None
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ScanResult:
    """All findings for one file plus the evidence used for grading."""

    file: str
    findings: List[Finding]
    subject_evidence: List[str]
    window_evidence: List[str]
    n_lines: int
    parse_error: Optional[str] = None

    @property
    def max_severity(self) -> str:
        best = "info"
        for f in self.findings:
            if SEVERITY_ORDER[f.severity] > SEVERITY_ORDER[best]:
                best = f.severity
        return best

    def rules(self) -> Set[str]:
        return {f.rule for f in self.findings}


@dataclass
class _Event:
    kind: str  # 'split' | 'tfit' | 'fsel' | 'resample' | 'efit' | 'eval' | 'metric' | 'search_fit'
    lineno: int
    names: Set[str]
    outputs: Set[str] = field(default_factory=set)
    group_aware: bool = False
    callee: str = ""
    kwargs: Dict[str, Set[str]] = field(default_factory=dict)
    cls: str = ""
    scope: str = "<module>"  # innermost enclosing function; order-based rules apply within one scope


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _callee_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _root_names(node: ast.AST) -> Set[str]:
    """Root variable names referenced by an expression (``X[:, 1]`` -> {X}, ``df.values`` -> {df})."""
    out: Set[str] = set()
    if node is None:
        return out
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, ast.Attribute):
        out |= _root_names(node.value)
    elif isinstance(node, ast.Subscript):
        out |= _root_names(node.value)
    elif isinstance(node, ast.Call):
        for a in node.args:
            out |= _root_names(a)
        for k in node.keywords:
            out |= _root_names(k.value)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            out |= _root_names(e)
    elif isinstance(node, ast.Starred):
        out |= _root_names(node.value)
    elif isinstance(node, ast.BinOp):
        out |= _root_names(node.left) | _root_names(node.right)
    elif isinstance(node, ast.UnaryOp):
        out |= _root_names(node.operand)
    return out


def _assign_targets(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            out |= _assign_targets(e)
    elif isinstance(node, ast.Starred):
        out |= _assign_targets(node.value)
    elif isinstance(node, (ast.Attribute, ast.Subscript)):
        out |= _root_names(node)
    return out


def _receiver_class(func: ast.AST, bindings: Dict[str, str]) -> Tuple[str, Set[str]]:
    """For ``recv.method(...)`` return ``(class_name, receiver_root_names)``.

    ``class_name`` is resolved from a direct constructor (``StandardScaler().fit``)
    or from a previous assignment (``sc = StandardScaler(); sc.fit``).
    """
    if not isinstance(func, ast.Attribute):
        return "", set()
    base = func.value
    if isinstance(base, ast.Call):
        return _callee_name(base.func), set()
    roots = _root_names(base)
    for r in roots:
        if r in bindings:
            return bindings[r], roots
    return "", roots


def _is_test_name(name: str) -> bool:
    low = name.lower()
    if low in {"pytest", "unittest", "latest", "attest", "contest", "test"}:
        return low == "test"
    return bool(TEST_NAME_RE.search(name))


def _iter_nodes_in_order(tree: ast.AST) -> Iterator[ast.AST]:
    nodes = [n for n in ast.walk(tree) if hasattr(n, "lineno")]
    nodes.sort(key=lambda n: (n.lineno, getattr(n, "col_offset", 0)))
    return iter(nodes)


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
class _Collector:
    def __init__(self, tree: ast.AST, source: str) -> None:
        self.tree = tree
        self.source = source
        self.bindings: Dict[str, str] = {}  # var name -> class name
        self.events: List[_Event] = []
        self.group_split_seen = False
        self.groups_kw_seen = False
        self.pipeline_seen = False
        self.identifiers: Set[str] = set()
        self.strings: List[str] = []
        self.func_ranges: List[Tuple[int, int, str]] = []

    def scope_of(self, lineno: int) -> str:
        """Innermost function whose line range contains ``lineno`` (``'<module>'`` if none)."""
        best: Optional[Tuple[int, int, str]] = None
        for lo, hi, name in self.func_ranges:
            if lo <= lineno <= hi and (best is None or (hi - lo) < (best[1] - best[0])):
                best = (lo, hi, name)
        return best[2] if best else "<module>"

    def run(self) -> None:
        # pass 0: function scopes (methods of the same class are different scopes)
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.func_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno, node.name))
        # pass 1: bindings and vocabulary
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                self.identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                self.identifiers.add(node.attr)
            elif isinstance(node, ast.arg):
                self.identifiers.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.strings.append(node.value[:200])
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    self.identifiers.add(alias.name.split(".")[-1])
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.identifiers.add(node.module)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                cls = _callee_name(node.value.func)
                if cls in TRANSFORMERS | FEATURE_SELECTORS | RESAMPLERS | ESTIMATORS | SEARCH_CLASSES | RANDOM_SPLITTERS | GROUP_SPLITTERS | PIPELINES:
                    for t in _assign_targets(node.targets[0]):
                        self.bindings[t] = cls
        # pass 2: events in source order
        parent_assign: Dict[int, Set[str]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                parent_assign[id(node.value)] = _assign_targets(node.targets[0])
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call) and node.target is not None:
                parent_assign[id(node.value)] = _assign_targets(node.target)
        for node in _iter_nodes_in_order(self.tree):
            if isinstance(node, ast.Call):
                n_before = len(self.events)
                self._visit_call(node, parent_assign.get(id(node), set()))
                for ev in self.events[n_before:]:
                    ev.scope = self.scope_of(node.lineno)

    # .................................................................... #
    def _kwargs(self, node: ast.Call) -> Dict[str, Set[str]]:
        return {k.arg: _root_names(k.value) for k in node.keywords if k.arg}

    def _visit_call(self, node: ast.Call, outputs: Set[str]) -> None:
        name = _callee_name(node.func)
        kwargs = self._kwargs(node)
        argnames: Set[str] = set()
        for a in node.args:
            argnames |= _root_names(a)
        for v in kwargs.values():
            argnames |= v
        has_groups = "groups" in kwargs
        if has_groups:
            self.groups_kw_seen = True
        cv_is_group = False
        for k in node.keywords:
            if k.arg == "cv" and isinstance(k.value, ast.Call) and _callee_name(k.value.func) in GROUP_SPLITTERS:
                cv_is_group = True
            if k.arg == "cv" and isinstance(k.value, ast.Name) and self.bindings.get(k.value.id) in GROUP_SPLITTERS:
                cv_is_group = True

        if name in PIPELINES:
            self.pipeline_seen = True

        # ---- splitter constructors / functions ----
        if name in GROUP_SPLITTERS:
            self.group_split_seen = True
            self.events.append(_Event("split", node.lineno, argnames, outputs, True, name, kwargs))
            return
        if name in RANDOM_SPLITTERS:
            if name == "train_test_split" or name == "random_split":
                self.events.append(_Event("split", node.lineno, argnames, outputs, has_groups, name, kwargs))
            else:
                # constructor only; the .split call is the real event, but record the constructor too
                self.events.append(_Event("split", node.lineno, argnames, outputs, has_groups, name, kwargs))
            return
        if name in CV_FUNCS:
            self.events.append(_Event("split", node.lineno, argnames, outputs, has_groups or cv_is_group, name, kwargs))
            if has_groups or cv_is_group:
                self.group_split_seen = True
            return
        if name in SEARCH_CLASSES:
            if cv_is_group:
                self.group_split_seen = True
            self.events.append(_Event("split", node.lineno, argnames, outputs, has_groups or cv_is_group, name, kwargs))
            return

        # ---- method calls on bound objects ----
        cls, recv = _receiver_class(node.func, self.bindings)
        if name == "split" and isinstance(node.func, ast.Attribute):
            if cls in GROUP_SPLITTERS or has_groups:
                self.group_split_seen = True
                self.events.append(_Event("split", node.lineno, argnames, outputs, True, f"{cls}.split", kwargs, cls))
            elif cls in RANDOM_SPLITTERS or recv & {"kf", "skf", "cv", "kfold", "folds", "splitter", "ss", "sss"}:
                self.events.append(_Event("split", node.lineno, argnames, outputs, False, f"{cls or 'unknown'}.split", kwargs, cls))
            return
        if name in FIT_METHODS and isinstance(node.func, ast.Attribute):
            if cls in FEATURE_SELECTORS:
                self.events.append(_Event("fsel", node.lineno, argnames, outputs, False, name, kwargs, cls))
            elif cls in TRANSFORMERS:
                self.events.append(_Event("tfit", node.lineno, argnames, outputs, False, name, kwargs, cls))
            elif cls in RESAMPLERS or name == "fit_resample":
                self.events.append(_Event("resample", node.lineno, argnames, outputs, False, name, kwargs, cls))
            elif cls in SEARCH_CLASSES:
                self.events.append(_Event("search_fit", node.lineno, argnames, outputs, cv_is_group, name, kwargs, cls))
            elif cls in PIPELINES:
                self.events.append(_Event("efit", node.lineno, argnames, outputs, False, name, kwargs, cls))
            elif name == "fit_transform":
                # unknown transformer (custom); still a transformer fit
                self.events.append(_Event("tfit", node.lineno, argnames, outputs, False, name, kwargs, cls or "unknown"))
            else:
                self.events.append(_Event("efit", node.lineno, argnames, outputs, False, name, kwargs, cls or "unknown"))
            return
        if name in EVAL_METHODS and isinstance(node.func, ast.Attribute):
            if name == "transform" and cls not in TRANSFORMERS | FEATURE_SELECTORS and cls != "":
                return
            self.events.append(_Event("eval", node.lineno, argnames, outputs, False, name, kwargs, cls))
            return
        if name in METRIC_FUNCS:
            self.events.append(_Event("metric", node.lineno, argnames, outputs, False, name, kwargs))


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #
def _evidence_tokens(regex: re.Pattern, identifiers: Iterable[str], strings: Iterable[str]) -> List[str]:
    found: Set[str] = set()
    for ident in identifiers:
        m = regex.search(ident)
        if m:
            found.add(m.group(0).lower())
    for s in strings:
        for m in regex.finditer(s):
            found.add(m.group(0).lower())
    return sorted(found)


def _grade(n_subject_tokens: int, forced: Optional[bool]) -> str:
    if forced is True:
        return "high"
    if forced is False:
        return "low"
    if n_subject_tokens >= 2:
        return "high"
    if n_subject_tokens == 1:
        return "medium"
    return "low"


def _apply_rules(col: _Collector, subject_tokens: List[str], window_tokens: List[str], forced: Optional[bool]) -> List[Finding]:
    findings: List[Finding] = []
    events = col.events
    splits = [e for e in events if e.kind in ("split", "search_fit")]
    random_splits = [e for e in splits if not e.group_aware]
    any_split = bool(splits)
    grade = _grade(len(subject_tokens), forced)

    # positive evidence
    if col.group_split_seen or col.groups_kw_seen:
        ln = min((e.lineno for e in splits if e.group_aware), default=0)
        findings.append(Finding("group-aware-split-present", "info", ln, "Group-aware splitter or groups= argument found.", evidence={"subject_tokens": subject_tokens}))
    if col.pipeline_seen:
        findings.append(Finding("pipeline-present", "info", 0, "sklearn Pipeline/ColumnTransformer used (preprocessing fitted inside CV folds if used with cross-validation)."))

    # R1 / R2: random split without groups
    if random_splits and not (col.group_split_seen or col.groups_kw_seen):
        first = min(random_splits, key=lambda e: e.lineno)
        msg = (
            f"Random split `{first.callee}` without groups=; file shows subject-level structure "
            f"({', '.join(subject_tokens) or 'no subject tokens found'}). Records/windows of one subject may land in both train and test."
        )
        findings.append(
            Finding(
                "split-without-groups",
                grade,
                first.lineno,
                msg,
                evidence={"subject_tokens": subject_tokens, "callee": first.callee, "n_random_splits": len(random_splits)},
            )
        )
        if window_tokens:
            findings.append(
                Finding(
                    "window-level-random-split",
                    grade,
                    first.lineno,
                    f"Random split on windowed/epoched data ({', '.join(window_tokens[:5])}); overlapping or adjacent windows of the same recording will be shared across folds.",
                    evidence={"window_tokens": window_tokens, "subject_tokens": subject_tokens},
                )
            )
    elif random_splits and (col.group_split_seen or col.groups_kw_seen):
        first = min(random_splits, key=lambda e: e.lineno)
        findings.append(
            Finding(
                "mixed-splitting",
                "low",
                first.lineno,
                f"Both group-aware and random splits (`{first.callee}`) appear; verify the random split is not used for the reported result.",
                evidence={"callee": first.callee},
            )
        )

    # R3-R5: preprocessing / feature selection / resampling before split
    def _before_split(ev: _Event) -> Optional[_Event]:
        touched = ev.names | ev.outputs
        # ignore fits on train-named arrays
        if ev.names and all(TRAIN_NAME_RE.search(n) for n in ev.names):
            return None
        # only compare order within the same function scope (or module level)
        later = [s for s in splits if s.lineno > ev.lineno and s.scope == ev.scope and (s.names & touched)]
        if later:
            return min(later, key=lambda e: e.lineno)
        return None

    seen_rules: Set[Tuple[str, int]] = set()
    for ev in events:
        if ev.kind not in ("tfit", "fsel", "resample"):
            continue
        sp = _before_split(ev)
        if sp is None:
            continue
        rule = {"tfit": "preprocess-before-split", "fsel": "feature-selection-before-split", "resample": "resample-before-split"}[ev.kind]
        key = (rule, ev.lineno)
        if key in seen_rules:
            continue
        seen_rules.add(key)
        sev = "high" if ev.kind in ("fsel", "resample") else "medium"
        findings.append(
            Finding(
                rule,
                sev,
                ev.lineno,
                f"`{ev.cls or 'transformer'}.{ev.callee}` on {sorted(ev.names)} at line {ev.lineno} precedes the split `{sp.callee}` at line {sp.lineno} that uses the same data.",
                evidence={"transformer": ev.cls, "names": sorted(ev.names), "split_line": sp.lineno},
            )
        )

    # R6: test set used in training / model selection
    for ev in events:
        if ev.kind not in ("efit", "search_fit", "tfit", "fsel", "resample"):
            continue
        test_pos = sorted(n for n in ev.names if _is_test_name(n))
        risky_kw = {k: sorted(v) for k, v in ev.kwargs.items() if k in ("validation_data", "eval_set", "validation_set", "X_val", "valid_sets") and any(_is_test_name(n) for n in v)}
        if ev.kind in ("tfit", "fsel", "resample"):
            # fitting a scaler on the test set is also leakage (should be transform only)
            if test_pos and ev.callee in ("fit", "fit_transform", "fit_resample"):
                findings.append(
                    Finding(
                        "test-set-in-training",
                        "high",
                        ev.lineno,
                        f"`{ev.cls or 'transformer'}.{ev.callee}` is fitted on test-named data {test_pos}; use transform() only.",
                        evidence={"names": test_pos},
                    )
                )
            continue
        if test_pos or risky_kw:
            what = "positional args " + str(test_pos) if test_pos else "keyword " + str(risky_kw)
            findings.append(
                Finding(
                    "test-set-in-training",
                    "high",
                    ev.lineno,
                    f"`{ev.callee}` receives test-named data via {what}; the test set influences training, early stopping or model selection.",
                    evidence={"names": test_pos, "kwargs": risky_kw, "callee": ev.callee},
                )
            )

    # R7: no hold-out at all
    if not any_split:
        efits = [e for e in events if e.kind == "efit" and e.names]
        evals = [e for e in events if e.kind in ("eval", "metric") and e.names]
        for ef in efits:
            same = [ev for ev in evals if ev.lineno > ef.lineno and ev.scope == ef.scope and (ev.names & ef.names) and not any(_is_test_name(n) for n in ev.names)]
            if same:
                findings.append(
                    Finding(
                        "no-holdout-evaluation",
                        "medium",
                        same[0].lineno,
                        f"Estimator fitted on {sorted(ef.names)} (line {ef.lineno}) and evaluated on the same arrays at line {same[0].lineno}; no split or cross-validation found in this file.",
                        evidence={"fit_line": ef.lineno, "names": sorted(ef.names & same[0].names)},
                    )
                )
                break
    return findings


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def scan_source(source: str, filename: str = "<string>", subject_level: Optional[bool] = None, line_map: Optional[NotebookSource] = None) -> ScanResult:
    """Scan Python source text and return a :class:`ScanResult`.

    Parameters
    ----------
    source:
        Python code.
    filename:
        Used only for reporting.
    subject_level:
        Force the subject-level assumption (``True`` -> grade random splits as
        ``high`` regardless of vocabulary; ``False`` -> ``low``).  ``None`` lets
        the vocabulary heuristics decide.
    line_map:
        Optional :class:`~leakscan.notebooks.NotebookSource` for cell lookup.
    """
    n_lines = source.count("\n") + 1
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - exercised via notebooks
        return ScanResult(filename, [Finding("parse-error", "info", exc.lineno or 0, f"SyntaxError: {exc.msg}", file=filename)], [], [], n_lines, parse_error=str(exc))
    col = _Collector(tree, source)
    col.run()
    subject_tokens = _evidence_tokens(SUBJECT_TOKEN_RE, col.identifiers, col.strings)
    window_tokens = _evidence_tokens(WINDOW_TOKEN_RE, col.identifiers, [])
    findings = _apply_rules(col, subject_tokens, window_tokens, subject_level)
    for f in findings:
        f.file = filename
        if line_map is not None and f.lineno:
            f.cell, f.cell_line = line_map.locate(f.lineno)
    findings.sort(key=lambda f: (-SEVERITY_ORDER[f.severity], f.lineno))
    return ScanResult(filename, findings, subject_tokens, window_tokens, n_lines)


def scan_file(path: str | Path, subject_level: Optional[bool] = None) -> ScanResult:
    """Scan a ``.py`` or ``.ipynb`` file."""
    p = Path(path)
    if p.suffix == ".ipynb":
        nbsrc = notebook_to_source(p)
        res = scan_source(nbsrc.source, str(p), subject_level, line_map=nbsrc)
        if res.parse_error:
            # fall back to per-cell parsing so one bad cell does not hide the rest
            return _scan_notebook_per_cell(p, subject_level)
        return res
    text = p.read_text(encoding="utf-8", errors="replace")
    return scan_source(text, str(p), subject_level)


def _scan_notebook_per_cell(p: Path, subject_level: Optional[bool]) -> ScanResult:
    import json

    nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    findings: List[Finding] = []
    subj: Set[str] = set()
    win: Set[str] = set()
    n_lines = 0
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        src = "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.split("\n"))
        r = scan_source(src, str(p), subject_level)
        n_lines += r.n_lines
        subj |= set(r.subject_evidence)
        win |= set(r.window_evidence)
        for f in r.findings:
            f.cell = ci
            f.cell_line = f.lineno
            findings.append(f)
    findings.append(Finding("parse-error", "info", 0, "Notebook scanned cell-by-cell (concatenated source did not parse); cross-cell ordering rules were not applied.", file=str(p)))
    return ScanResult(str(p), findings, sorted(subj), sorted(win), n_lines, parse_error="per-cell")


SKIP_DIRS = {".git", "node_modules", "site-packages", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", "build", "dist", ".tox", ".mypy_cache"}


def iter_code_files(root: str | Path, max_bytes: int = 2_000_000) -> Iterator[Path]:
    """Yield ``.py`` and ``.ipynb`` files under ``root`` skipping vendored/virtualenv dirs."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in (".py", ".ipynb") and p.is_file():
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield p


def scan_path(root: str | Path, subject_level: Optional[bool] = None) -> List[ScanResult]:
    """Scan every code file under ``root`` (a file or directory)."""
    return [scan_file(p, subject_level) for p in iter_code_files(root)]
