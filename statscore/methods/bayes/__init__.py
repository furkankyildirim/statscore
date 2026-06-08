"""Bayesian inference: conjugate priors and MCMC."""

from __future__ import annotations

from statscore.methods.bayes._mcmc_results import ConjugateModelResult, MCMCResult
from statscore.methods.bayes._results import NormalMeanKnownVarResult, NormalMeanUnknownVarResult
from statscore.methods.bayes.conjugate import (
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
)
from statscore.methods.bayes.mcmc import (
    bayes_beta_binomial,
    bayes_gamma_poisson,
    mcmc_linear_regression,
    mcmc_normal_mean_unknown_var,
    run_mcmc,
)

__all__ = [
    # Conjugate closed-form
    "bayes_normal_mean_known_var",
    "bayes_normal_mean_unknown_var",
    "NormalMeanKnownVarResult",
    "NormalMeanUnknownVarResult",
    # MCMC
    "run_mcmc",
    "mcmc_normal_mean_unknown_var",
    "mcmc_linear_regression",
    "MCMCResult",
    # New conjugate models
    "bayes_beta_binomial",
    "bayes_gamma_poisson",
    "ConjugateModelResult",
]
