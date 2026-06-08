"""Bayesian inference with conjugate priors for normal data."""

from __future__ import annotations

from statscore.bayes.conjugate import (
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
)

__all__ = [
    "bayes_normal_mean_known_var",
    "bayes_normal_mean_unknown_var",
]
