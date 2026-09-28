"""nf_multiverse - specification-curve re-analysis of the *feedback signal* in EEG neurofeedback.

Modules
-------
- :mod:`nf_multiverse.specs` - the specification space (reference, band, estimator,
  window, normalisation, artifact handling, smoothing) as frozen dataclasses.
- :mod:`nf_multiverse.feedback` - recompute the feedback time series and the
  reward decisions from raw EEG under any specification; agreement metrics.
- :mod:`nf_multiverse.learning` - learning indices (within-session and
  session-to-session slopes), learner classification, classification stability
  (kappa), specification curves and the joint permutation test.
- :mod:`nf_multiverse.artifacts` - how much of a feedback signal is explained by
  ocular / muscular regressors (feedback specificity) and how contingent the
  delivered feedback is on the intended neural target.
- :mod:`nf_multiverse.synthetic` - synthetic multichannel EEG with a learning
  alpha oscillator, blinks and EMG bursts, for tests and power analyses.
"""
from .artifacts import artifact_regressors, contingency, feedback_specificity
from .feedback import FeedbackResult, compute_feedback, decision_agreement, individual_alpha_frequency, reward_decisions
from .learning import classify_learner, cohen_kappa, learner_agreement, specification_curve, specification_curve_test, within_session_slope
from .specs import DEFAULT_LEVELS, FeedbackSpec, build_specification_space, spec_table
from .synthetic import SimulatedSession, simulate_session

__all__ = [
    "FeedbackSpec",
    "DEFAULT_LEVELS",
    "build_specification_space",
    "spec_table",
    "FeedbackResult",
    "compute_feedback",
    "reward_decisions",
    "decision_agreement",
    "individual_alpha_frequency",
    "within_session_slope",
    "classify_learner",
    "cohen_kappa",
    "learner_agreement",
    "specification_curve",
    "specification_curve_test",
    "artifact_regressors",
    "feedback_specificity",
    "contingency",
    "SimulatedSession",
    "simulate_session",
]

__version__ = "0.1.0"
