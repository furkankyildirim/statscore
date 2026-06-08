"""Metropolis-Hastings MCMC sampler and non-conjugate Bayesian models.

Supported models via run_mcmc():
  - Normal mean (unknown mu, unknown sigma^2): flat or normal prior on mu,
    inverse-gamma (via log-normal proposal) or flat prior on sigma^2.
  - Any user-supplied log-posterior function.

Conjugate closed-form models (separate functions):
  - bayes_beta_binomial : Beta prior on p, Binomial likelihood
  - bayes_gamma_poisson : Gamma prior on lambda, Poisson likelihood
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from statscore.methods.bayes._mcmc_results import ConjugateModelResult, MCMCResult
from statscore.utils.validation import validate_1d_sample, validate_positive

# ---------------------------------------------------------------------------
# Metropolis-Hastings engine
# ---------------------------------------------------------------------------

def _mh_sampler(
    log_posterior: Callable[[np.ndarray], float],
    init: np.ndarray,
    n_iter: int,
    proposal_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Run Metropolis-Hastings with Gaussian random-walk proposals.

    Parameters
    ----------
    log_posterior : callable
        Function mapping parameter vector → log-posterior (up to additive const).
    init : array of shape (p,)
        Initial parameter values.
    n_iter : int
        Total number of iterations.
    proposal_std : array of shape (p,)
        Standard deviations for Gaussian proposal on each parameter.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    chain : array of shape (n_iter, p)
    acceptance_rate : float
    """
    p = len(init)
    chain = np.empty((n_iter, p))
    current = init.copy()
    current_lp = log_posterior(current)
    accepted = 0

    for t in range(n_iter):
        proposal = current + rng.normal(0.0, proposal_std, size=p)
        prop_lp = log_posterior(proposal)
        log_alpha = prop_lp - current_lp
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_lp = prop_lp
            accepted += 1
        chain[t] = current

    return chain, accepted / n_iter


def run_mcmc(
    log_posterior: Callable[[np.ndarray], float],
    init: np.ndarray,
    param_names: list[str],
    n_iter: int = 10_000,
    n_burnin: int = 2_000,
    proposal_std: np.ndarray | None = None,
    alpha: float = 0.05,
    model_name: str = "Custom Model",
    seed: int = 42,
) -> MCMCResult:
    """Run Metropolis-Hastings MCMC for any model with a log-posterior function.

    Parameters
    ----------
    log_posterior : callable
        Function (params: np.ndarray) -> float giving log-posterior value
        (proportional; additive constants are fine).
    init : array-like
        Initial parameter values, shape (p,).
    param_names : list of str
        Names for each parameter.
    n_iter : int
        Total number of MH iterations (including burn-in).
    n_burnin : int
        Number of initial iterations to discard.
    proposal_std : array-like or None
        Proposal standard deviations for each parameter.
        Defaults to 0.5 for all parameters.
    alpha : float
        Credible interval level (1 - alpha).
    model_name : str
        Label shown in summary/plot.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MCMCResult
    """
    init = np.asarray(init, dtype=float)
    p = len(init)
    if len(param_names) != p:
        raise ValueError("len(param_names) must match len(init).")
    if proposal_std is None:
        proposal_std = np.full(p, 0.5)
    else:
        proposal_std = np.asarray(proposal_std, dtype=float)

    validate_positive(alpha, "alpha")
    if n_burnin >= n_iter:
        raise ValueError("n_burnin must be less than n_iter.")

    rng = np.random.default_rng(seed)
    chain, acc_rate = _mh_sampler(log_posterior, init, n_iter, proposal_std, rng)

    post_samples = chain[n_burnin:]

    post_mean = post_samples.mean(axis=0)
    post_std = post_samples.std(axis=0, ddof=1)

    q_lo = alpha / 2 * 100
    q_hi = (1 - alpha / 2) * 100
    cis = [
        (float(np.percentile(post_samples[:, i], q_lo)),
         float(np.percentile(post_samples[:, i], q_hi)))
        for i in range(p)
    ]

    return MCMCResult(
        chain=chain,
        param_names=list(param_names),
        n_iter=n_iter,
        n_burnin=n_burnin,
        acceptance_rate=acc_rate,
        posterior_samples=post_samples,
        posterior_mean=post_mean,
        posterior_std=post_std,
        credible_intervals=cis,
        alpha=alpha,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Built-in MCMC models for normal data
# ---------------------------------------------------------------------------

def mcmc_normal_mean_unknown_var(
    x: np.ndarray,
    mu_prior_mean: float = 0.0,
    mu_prior_std: float = 10.0,
    sigma_prior_alpha: float = 2.0,
    sigma_prior_beta: float = 1.0,
    n_iter: int = 12_000,
    n_burnin: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> MCMCResult:
    """MCMC for normal mean and variance with conjugacy-breaking priors.

    Prior:
        mu    ~ Normal(mu_prior_mean, mu_prior_std^2)   [can be diffuse]
        sigma ~ InvGamma(sigma_prior_alpha, sigma_prior_beta)

    This implements MCMC even though a conjugate solution also exists,
    so the user can see how MH compares against the closed-form result
    and can change priors freely.

    Parameters
    ----------
    x : array-like
        Observations.
    mu_prior_mean : float
        Mean of the Normal prior on mu.
    mu_prior_std : float
        Std of the Normal prior on mu. Large value → diffuse.
    sigma_prior_alpha : float
        Shape of the Inverse-Gamma prior on sigma^2.
    sigma_prior_beta : float
        Scale of the Inverse-Gamma prior on sigma^2.
    n_iter : int
        Total MH iterations.
    n_burnin : int
        Burn-in iterations to discard.
    alpha : float
        Credible interval level.
    seed : int
        RNG seed.

    Returns
    -------
    MCMCResult  with param_names = ["mu", "log_sigma"]
        Note: the chain is in (mu, log_sigma) space for unconstrained sampling.
        Posterior samples for sigma = exp(log_sigma).
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x, min_obs=2)
    validate_positive(mu_prior_std, "mu_prior_std")
    validate_positive(sigma_prior_alpha, "sigma_prior_alpha")
    validate_positive(sigma_prior_beta, "sigma_prior_beta")

    n = len(x)
    x_bar = float(x.mean())
    ss = float(np.sum((x - x_bar) ** 2))

    # Work in (mu, log_sigma) — ensures sigma > 0 without constrained proposals.
    # log|J| = log_sigma is the Jacobian for the change of variables.
    def log_posterior(params: np.ndarray) -> float:
        mu, log_sigma = float(params[0]), float(params[1])
        sigma = np.exp(log_sigma)
        if sigma <= 0:
            return -np.inf
        # Log-likelihood N(mu, sigma^2)
        ll = -n * log_sigma - 0.5 * (ss + n * (x_bar - mu) ** 2) / sigma**2
        # Prior on mu: N(mu_prior_mean, mu_prior_std^2)
        lp_mu = -0.5 * ((mu - mu_prior_mean) / mu_prior_std) ** 2
        # Prior on sigma^2: InvGamma(alpha, beta) → log p(sigma^2) + Jacobian
        # log p(sigma^2) = alpha*log(beta) - lgamma(alpha) - (alpha+1)*log(sigma^2) - beta/sigma^2
        log_sigma2 = 2 * log_sigma
        lp_sig = (sigma_prior_alpha * np.log(sigma_prior_beta)
                  - (sigma_prior_alpha + 1) * log_sigma2
                  - sigma_prior_beta / sigma**2)
        # Jacobian: d(log_sigma)/d(sigma) term already in lp_sig via log_sigma2
        # plus one more log_sigma for log|d sigma^2 / d log_sigma| = 2 log_sigma
        return float(ll + lp_mu + lp_sig + log_sigma)

    # Initialise at sample estimates
    init = np.array([x_bar, float(np.log(x.std(ddof=1) + 1e-6))])
    proposal_std = np.array([float(x.std(ddof=1)) / np.sqrt(n) * 2.0, 0.3])

    result = run_mcmc(
        log_posterior=log_posterior,
        init=init,
        param_names=["mu", "log_sigma"],
        n_iter=n_iter,
        n_burnin=n_burnin,
        proposal_std=proposal_std,
        alpha=alpha,
        model_name="Normal(mu, sigma²) — MCMC",
        seed=seed,
    )
    # Add sigma on the result as extra info via a secondary result with exp transform
    sigma_samples = np.exp(result.posterior_samples[:, 1])
    sigma_mean = float(sigma_samples.mean())
    sigma_std = float(sigma_samples.std(ddof=1))
    sigma_ci = (
        float(np.percentile(sigma_samples, alpha / 2 * 100)),
        float(np.percentile(sigma_samples, (1 - alpha / 2) * 100)),
    )

    # Extend result with sigma info
    result.param_names = ["mu", "log_sigma (→sigma)"]
    result.posterior_mean[1] = sigma_mean
    result.posterior_std[1] = sigma_std
    result.credible_intervals[1] = sigma_ci
    # Re-label for clarity
    result.param_names = ["mu", "sigma"]

    return result


def mcmc_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    beta_prior_mean: np.ndarray | None = None,
    beta_prior_std: float = 10.0,
    sigma_prior_alpha: float = 2.0,
    sigma_prior_beta: float = 1.0,
    n_iter: int = 15_000,
    n_burnin: int = 3_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> MCMCResult:
    """MCMC for normal linear regression Y = X beta + eps, eps ~ N(0, sigma^2).

    Prior:
        beta_j ~ Normal(beta_prior_mean[j], beta_prior_std^2)  (independent)
        sigma  ~ InvGamma(sigma_prior_alpha, sigma_prior_beta)

    Parameters
    ----------
    X : array of shape (n, p)
        Design matrix (include intercept column of ones if needed).
    y : array of shape (n,)
        Response vector.
    beta_prior_mean : array of shape (p,) or None
        Prior means for beta. Defaults to zeros.
    beta_prior_std : float
        Shared prior std for all beta_j.
    sigma_prior_alpha, sigma_prior_beta : float
        Inverse-Gamma prior parameters for sigma^2.
    n_iter, n_burnin : int
    alpha : float
    seed : int

    Returns
    -------
    MCMCResult with param_names = ["beta_0", "beta_1", ..., "sigma"]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must be 2-D with X.shape[0] == len(y).")
    n, p = X.shape
    if beta_prior_mean is None:
        beta_prior_mean_arr = np.zeros(p)
    else:
        beta_prior_mean_arr = np.asarray(beta_prior_mean, dtype=float)

    validate_positive(beta_prior_std, "beta_prior_std")
    validate_positive(sigma_prior_alpha, "sigma_prior_alpha")
    validate_positive(sigma_prior_beta, "sigma_prior_beta")

    # params = [beta_0, ..., beta_{p-1}, log_sigma]
    def log_posterior(params: np.ndarray) -> float:
        beta = params[:p]
        log_sigma = float(params[p])
        sigma = np.exp(log_sigma)
        resid = y - X @ beta
        ll = -n * log_sigma - 0.5 * np.dot(resid, resid) / sigma**2
        lp_beta = -0.5 * np.sum(((beta - beta_prior_mean_arr) / beta_prior_std) ** 2)
        log_sigma2 = 2 * log_sigma
        lp_sig = (sigma_prior_alpha * np.log(sigma_prior_beta)
                  - (sigma_prior_alpha + 1) * log_sigma2
                  - sigma_prior_beta / sigma**2)
        return float(ll + lp_beta + lp_sig + log_sigma)

    # OLS initialisation
    try:
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        resid_ols = y - X @ beta_ols
        sigma_ols = float(np.std(resid_ols, ddof=p))
    except Exception:
        beta_ols = np.zeros(p)
        sigma_ols = 1.0
    log_sigma_init = float(np.log(max(sigma_ols, 1e-6)))
    init = np.append(beta_ols, log_sigma_init)

    prop_beta_std = float(sigma_ols / np.sqrt(n)) * 3.0
    proposal_std = np.append(np.full(p, prop_beta_std), 0.3)

    param_names = [f"beta_{i}" for i in range(p)] + ["log_sigma_raw"]

    result = run_mcmc(
        log_posterior=log_posterior,
        init=init,
        param_names=param_names,
        n_iter=n_iter,
        n_burnin=n_burnin,
        proposal_std=proposal_std,
        alpha=alpha,
        model_name="Linear Regression — MCMC",
        seed=seed,
    )

    # Transform last parameter (log_sigma → sigma)
    sigma_samples = np.exp(result.posterior_samples[:, p])
    result.posterior_mean[p] = float(sigma_samples.mean())
    result.posterior_std[p] = float(sigma_samples.std(ddof=1))
    result.credible_intervals[p] = (
        float(np.percentile(sigma_samples, alpha / 2 * 100)),
        float(np.percentile(sigma_samples, (1 - alpha / 2) * 100)),
    )
    result.param_names = [f"beta_{i}" for i in range(p)] + ["sigma"]
    return result


# ---------------------------------------------------------------------------
# Closed-form conjugate models: Beta-Binomial and Gamma-Poisson
# ---------------------------------------------------------------------------

def bayes_beta_binomial(
    n_trials: int,
    n_successes: int,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    alpha: float = 0.05,
) -> ConjugateModelResult:
    """Bayesian Beta-Binomial model: Beta prior on success probability p.

    Model: X ~ Binomial(n, p),  prior p ~ Beta(alpha0, beta0).

    The posterior is p | x ~ Beta(alpha0 + x, beta0 + n - x).

    Parameters
    ----------
    n_trials : int
        Total number of trials.
    n_successes : int
        Observed number of successes.
    alpha0 : float
        Prior shape1 (pseudo-successes). alpha0 = beta0 = 1 → uniform prior.
    beta0 : float
        Prior shape2 (pseudo-failures).
    alpha : float
        Credible interval level (1 - alpha).

    Returns
    -------
    ConjugateModelResult
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if not (0 <= n_successes <= n_trials):
        raise ValueError("n_successes must be in [0, n_trials].")
    validate_positive(alpha0, "alpha0")
    validate_positive(beta0, "beta0")
    validate_positive(alpha, "alpha")

    from scipy import stats

    alpha_n = alpha0 + n_successes
    beta_n = beta0 + n_trials - n_successes

    post_mean = alpha_n / (alpha_n + beta_n)
    post_var = (alpha_n * beta_n) / ((alpha_n + beta_n) ** 2 * (alpha_n + beta_n + 1))
    post_std = float(np.sqrt(post_var))

    ci_lo = float(stats.beta.ppf(alpha / 2, alpha_n, beta_n))
    ci_hi = float(stats.beta.ppf(1 - alpha / 2, alpha_n, beta_n))

    return ConjugateModelResult(
        model_name="Beta-Binomial",
        prior_params={"alpha0": alpha0, "beta0": beta0},
        posterior_params={"alpha_n": alpha_n, "beta_n": beta_n},
        posterior_mean=float(post_mean),
        posterior_variance=float(post_var),
        posterior_std=post_std,
        credible_interval=(ci_lo, ci_hi),
        n=n_trials,
        alpha=alpha,
        data_summary={"successes": float(n_successes),
                      "MLE_p": float(n_successes / n_trials)},
    )


def bayes_gamma_poisson(
    x: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    alpha: float = 0.05,
) -> ConjugateModelResult:
    """Bayesian Gamma-Poisson model: Gamma prior on Poisson rate lambda.

    Model: X_i ~ Poisson(lambda),  prior lambda ~ Gamma(alpha0, beta0).
    (beta0 here is the *rate* parameter, so prior mean = alpha0 / beta0.)

    Posterior: lambda | x ~ Gamma(alpha0 + sum(x), beta0 + n).

    Parameters
    ----------
    x : array-like
        Non-negative integer counts.
    alpha0 : float
        Prior shape.
    beta0 : float
        Prior rate (= 1/scale). Large beta0 → prior concentrated near 0.
    alpha : float
        Credible interval level.

    Returns
    -------
    ConjugateModelResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x, min_obs=1)
    if np.any(x < 0):
        raise ValueError("Poisson data must be non-negative.")
    validate_positive(alpha0, "alpha0")
    validate_positive(beta0, "beta0")
    validate_positive(alpha, "alpha")

    from scipy import stats

    n = len(x)
    x_sum = float(x.sum())
    x_bar = float(x.mean())

    alpha_n = alpha0 + x_sum
    beta_n = beta0 + n

    post_mean = alpha_n / beta_n
    post_var = alpha_n / beta_n**2
    post_std = float(np.sqrt(post_var))

    ci_lo = float(stats.gamma.ppf(alpha / 2, alpha_n, scale=1.0 / beta_n))
    ci_hi = float(stats.gamma.ppf(1 - alpha / 2, alpha_n, scale=1.0 / beta_n))

    return ConjugateModelResult(
        model_name="Gamma-Poisson",
        prior_params={"alpha0": alpha0, "beta0": beta0},
        posterior_params={"alpha_n": alpha_n, "beta_n": beta_n},
        posterior_mean=float(post_mean),
        posterior_variance=float(post_var),
        posterior_std=post_std,
        credible_interval=(ci_lo, ci_hi),
        n=n,
        alpha=alpha,
        data_summary={"sum_x": x_sum, "x_bar": x_bar},
    )
