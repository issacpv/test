"""Specification grid, sepsis labelling per specification, cohort geometry and variance decomposition."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .sofa import SofaConfig, hourly_sofa, sofa_baseline
from .suspected_infection import SIRule, suspected_infection_times

DIMENSIONS = ["si_rule", "abx_list", "suspicion_time", "sofa_before_h", "sofa_after_h", "sofa_baseline",
              "sofa_delta", "carry_forward", "assume_normal", "onset_convention", "onset_relative_to_icu"]


@dataclass(frozen=True)
class SepsisSpec:
    """One point in the Sepsis-3 multiverse. Field names are the grid dimensions."""

    si_rule: str = "culture_abx"            # see suspected_infection.RULES
    abx_list: str = "mimic_code"            # "mimic_code" | "broad_j01"
    suspicion_time: str = "earlier"         # "earlier" | "antibiotic" | "culture"
    sofa_before_h: int = 48                 # window before suspicion (Seymour: 48)
    sofa_after_h: int = 24                  # window after suspicion (Seymour: 24)
    sofa_baseline: str = "zero"             # "zero" | "min_lookback" | "first_icu" | "min_prior"
    sofa_delta: int = 2
    carry_forward: str = "default"          # "default" | "none" | "long"
    assume_normal: bool = True
    onset_convention: str = "suspicion"     # "suspicion" | "sofa_rise"
    onset_relative_to_icu: str = "any"      # "any" | "after_admission" | "within_24h_before"

    def name(self) -> str:
        return "|".join(f"{k}={v}" for k, v in asdict(self).items())

    def sofa_config(self) -> SofaConfig:
        cfg = SofaConfig(assume_normal=self.assume_normal)
        if self.carry_forward == "none":
            cfg.carry_forward_h = {k: 0 for k in cfg.carry_forward_h}
        elif self.carry_forward == "long":
            cfg.carry_forward_h = {k: (72 if v > 1 else v) for k, v in cfg.carry_forward_h.items()}
        return cfg

    def si_rule_obj(self) -> SIRule:
        return SIRule(rule=self.si_rule, suspicion_time=self.suspicion_time)


REFERENCE_SPECS: Dict[str, SepsisSpec] = {
    # mimic-code sepsis3.sql: Seymour windows, baseline zero, onset = suspicion time
    "mimic_code": SepsisSpec(),
    # ricu / YAIB style: suspicion from antibiotics + culture, SOFA rise within [-48, +24], baseline = minimum before
    "ricu_like": SepsisSpec(sofa_baseline="min_prior", onset_convention="sofa_rise"),
    # AI Clinician (Komorowski 2018) used an antibiotic/culture pairing with a [-24, +48] style window in MIMIC-III
    "ai_clinician_like": SepsisSpec(sofa_before_h=24, sofa_after_h=48),
    # eICU fallback: antibiotic-only suspicion (culture timing sparse)
    "eicu_abx_only": SepsisSpec(si_rule="abx_only", suspicion_time="antibiotic"),
}


def default_grid() -> List[SepsisSpec]:
    """Fully crossed main grid (288 specifications).

    Dimensions: suspected-infection rule (3) x suspicion-time convention (2; collapsed for the
    antibiotic-only rule) x SOFA window (3) x SOFA baseline (3) x missingness handling (2:
    lenient = carry-forward + assume-normal, strict = neither) x onset convention (2) x onset
    position relative to ICU admission (2). The antibiotic-list dimension and the remaining
    missingness combinations are run as sensitivity sets on the reference specs, not crossed.
    """
    levels = {
        "si_rule": ["culture_abx", "abx_only", "abx_2plus"],
        "suspicion_time": ["earlier", "antibiotic"],
        "sofa_window": [(48, 24), (24, 12), (72, 24)],
        "sofa_baseline": ["zero", "min_lookback", "first_icu"],
        "missingness": [("default", True), ("none", False)],
        "onset_convention": ["suspicion", "sofa_rise"],
        "onset_relative_to_icu": ["any", "after_admission"],
    }
    specs = []
    for si, st, (b, a), base, (cf, an), oc, rel in itertools.product(*levels.values()):
        if si in ("abx_only", "abx_2plus") and st == "earlier":
            continue  # no culture in the pair; "earlier" is identical to "antibiotic"
        specs.append(SepsisSpec(si_rule=si, suspicion_time=st, sofa_before_h=b, sofa_after_h=a, sofa_baseline=base,
                                carry_forward=cf, assume_normal=an, onset_convention=oc, onset_relative_to_icu=rel))
    return specs


def sensitivity_specs(base: SepsisSpec) -> List[SepsisSpec]:
    """Non-crossed sensitivity variants around one specification (antibiotic list, carry-forward length, delta)."""
    return [replace(base, abx_list="broad_j01"), replace(base, carry_forward="long"),
            replace(base, assume_normal=not base.assume_normal), replace(base, sofa_delta=3),
            replace(base, onset_relative_to_icu="within_24h_before")]


@dataclass
class SepsisLabels:
    onset_h: pd.Series          # onset hour per stay (NaN if not septic)
    t_si: pd.Series             # suspicion time per stay (NaN if none)
    sofa_at_onset: pd.Series
    baseline: pd.Series


def label_sepsis(hourly: pd.DataFrame, antibiotics: pd.DataFrame, cultures: Optional[pd.DataFrame],
                 spec: SepsisSpec, stays: Optional[Sequence] = None) -> SepsisLabels:
    """Sepsis-3 onset per stay for one specification.

    ``hourly`` is the (stay_id, hour)-indexed wide concept table (see sofa.py); hours may be
    negative (pre-ICU data). A stay is septic if, within [t_si - before, t_si + after], the
    hourly SOFA minus baseline reaches ``sofa_delta``. Onset is the suspicion time or the
    first hour of the SOFA rise, then filtered by ``onset_relative_to_icu``.
    """
    sofa = hourly_sofa(hourly, spec.sofa_config())["sofa"]
    t_si = suspected_infection_times(antibiotics, cultures, spec.si_rule_obj())
    all_stays = list(stays) if stays is not None else sorted(set(hourly.index.get_level_values(0)))
    onset, sofa_on, base_s = {}, {}, {}
    for sid in all_stays:
        onset[sid], sofa_on[sid], base_s[sid] = np.nan, np.nan, np.nan
        if sid not in t_si.index or not np.isfinite(t_si[sid]) or sid not in sofa.index.get_level_values(0):
            continue
        ts = float(t_si[sid])
        s = sofa.xs(sid, level=0)
        base = sofa_baseline(s, int(np.floor(ts)), spec.sofa_baseline)
        w = s[(s.index >= ts - spec.sofa_before_h) & (s.index <= ts + spec.sofa_after_h)]
        rise = w[(w - base) >= spec.sofa_delta]
        if rise.empty:
            continue
        t_on = ts if spec.onset_convention == "suspicion" else float(rise.index[0])
        if spec.onset_relative_to_icu == "after_admission" and t_on < 0:
            continue
        if spec.onset_relative_to_icu == "within_24h_before" and t_on < -24:
            continue
        onset[sid], sofa_on[sid], base_s[sid] = t_on, float(rise.iloc[0]), base
    idx = pd.Index(all_stays, name="stay_id")
    return SepsisLabels(onset_h=pd.Series(onset).reindex(idx), t_si=t_si.reindex(idx),
                        sofa_at_onset=pd.Series(sofa_on).reindex(idx), baseline=pd.Series(base_s).reindex(idx))


def run_grid(hourly: pd.DataFrame, antibiotics: pd.DataFrame, cultures: Optional[pd.DataFrame],
             specs: Iterable[SepsisSpec], mortality: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Label every specification; return a summary table and the per-spec onset series."""
    rows, onsets = [], {}
    for spec in specs:
        lab = label_sepsis(hourly, antibiotics, cultures, spec)
        septic = lab.onset_h.notna()
        row = asdict(spec)
        row.update({"n_septic": int(septic.sum()), "prevalence": float(septic.mean()) if len(septic) else np.nan,
                    "median_onset_h": float(lab.onset_h.median()) if septic.any() else np.nan,
                    "frac_onset_before_icu": float((lab.onset_h[septic] < 0).mean()) if septic.any() else np.nan})
        if mortality is not None:
            m = mortality.reindex(lab.onset_h.index)
            row["mortality_septic"] = float(m[septic].mean()) if septic.any() else np.nan
        rows.append(row)
        onsets[spec.name()] = lab.onset_h
    return pd.DataFrame(rows), onsets


def pairwise_jaccard(onsets: Dict[str, pd.Series]) -> pd.DataFrame:
    names = list(onsets)
    sets = {n: set(onsets[n].dropna().index) for n in names}
    J = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            u = len(sets[a] | sets[b])
            J[i, j] = J[j, i] = len(sets[a] & sets[b]) / u if u else np.nan
    return pd.DataFrame(J, index=names, columns=names)


def onset_shift(onsets_a: pd.Series, onsets_b: pd.Series) -> Dict[str, float]:
    """Onset-time difference (b - a) for stays septic under both specs."""
    both = onsets_a.notna() & onsets_b.notna()
    d = (onsets_b - onsets_a)[both]
    return {"n_both": int(both.sum()), "median_shift_h": float(d.median()) if len(d) else np.nan,
            "iqr_h": float(d.quantile(0.75) - d.quantile(0.25)) if len(d) else np.nan,
            "frac_moved_gt_6h": float((d.abs() > 6).mean()) if len(d) else np.nan}


def variance_decomposition(summary: pd.DataFrame, outcome: str, dimensions: Sequence[str] = DIMENSIONS,
                           interactions: bool = False) -> pd.DataFrame:
    """Eta^2 of each grid dimension (and optional pairwise interactions) for an outcome column.

    Sequential (type-I) sums of squares are order dependent, so main effects are computed as
    the SS explained by each dimension alone (one-way eta^2); interactions are the increment
    over the two main effects. With a fully crossed grid these are the orthogonal ANOVA
    components; for fractional grids they are approximate and are flagged in the output.
    """
    df = summary.dropna(subset=[outcome]).copy()
    y = df[outcome].to_numpy(float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    dims = [d for d in dimensions if d in df and df[d].nunique() > 1]
    rows = []
    for d in dims:
        m = df.groupby(d)[outcome].transform("mean").to_numpy()
        rows.append({"term": d, "eta2": float(np.sum((m - y.mean()) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
                     "n_levels": int(df[d].nunique())})
    if interactions:
        for a, b in itertools.combinations(dims, 2):
            m_a = df.groupby(a)[outcome].transform("mean").to_numpy()
            m_b = df.groupby(b)[outcome].transform("mean").to_numpy()
            m_ab = df.groupby([a, b])[outcome].transform("mean").to_numpy()
            ss = np.sum((m_ab - m_a - m_b + y.mean()) ** 2)
            rows.append({"term": f"{a}:{b}", "eta2": float(ss / ss_tot) if ss_tot > 0 else np.nan,
                         "n_levels": int(df.groupby([a, b]).ngroups)})
    out = pd.DataFrame(rows).sort_values("eta2", ascending=False).reset_index(drop=True)
    balanced = all(df[d].value_counts().nunique() == 1 for d in dims)
    out.attrs["balanced_grid"] = balanced
    return out


def specification_curve(summary: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Sort specifications by the outcome for a specification-curve plot (rank, outcome, dimensions)."""
    df = summary.sort_values(outcome).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
