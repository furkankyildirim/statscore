"""Bayesian inference with conjugate priors for normal data."""

from __future__ import annotations

from statscore.methods.bayes._results import NormalMeanKnownVarResult, NormalMeanUnknownVarResult
from statscore.methods.bayes.conjugate import (
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
)

__all__ = [
    "bayes_normal_mean_known_var",
    "bayes_normal_mean_unknown_var",
    "NormalMeanKnownVarResult",
    "NormalMeanUnknownVarResult",
]
