"""meg_microstates: heritability and cardiac coupling of resting MEG microstate dynamics.

Modules
-------
microstates  : GFP, GFP-peak maps, modified k-means, back-fitting, smoothing, temporal
               parameters (duration, occurrence, coverage, transitions, entropy, DFA exponent).
hrv          : R-peak detection, RR cleaning, time/frequency-domain HRV, cardiac-phase
               histograms of microstate transitions.
heritability : twin ICCs, Falconer estimates, ACE/AE/CE/E maximum likelihood with LRTs,
               bootstrap CIs, twin identification, simulation-based power.
synthetic    : synthetic multichannel microstate data, ECG, twin phenotypes.
"""
from .microstates import (
    gfp,
    gfp_peaks,
    modified_kmeans,
    backfit,
    smooth_labels,
    microstate_parameters,
    global_explained_variance,
    dfa_exponent,
)
from .hrv import detect_r_peaks, rr_intervals, clean_rr, time_domain_hrv, frequency_domain_hrv, cardiac_phase_histogram
from .heritability import twin_icc, falconer, fit_ace, ace_model_comparison, twin_identification, power_simulation
from .synthetic import simulate_microstate_data, simulate_ecg, simulate_twin_phenotypes

__all__ = [
    "gfp", "gfp_peaks", "modified_kmeans", "backfit", "smooth_labels", "microstate_parameters",
    "global_explained_variance", "dfa_exponent",
    "detect_r_peaks", "rr_intervals", "clean_rr", "time_domain_hrv", "frequency_domain_hrv", "cardiac_phase_histogram",
    "twin_icc", "falconer", "fit_ace", "ace_model_comparison", "twin_identification", "power_simulation",
    "simulate_microstate_data", "simulate_ecg", "simulate_twin_phenotypes",
]
__version__ = "0.1.0"
