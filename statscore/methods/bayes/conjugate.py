"""Bayesian conjugate prior inference for normal data."""

from __future__ import annotations

import numpy as np

from statscore.methods.bayes import NormalMeanKnownVarResult, NormalMeanUnknownVarResult
from statscore.utils.distributions import chi2_ppf, norm_ppf, t_ppf
from statscore.utils.validation import validate_1d_sample, validate_positive


def bayes_normal_mean_known_var(
    x: np.ndarray,
    sigma_sq: float,
    mu0: float,
    kappa0: float,
    alpha: float = 0.05,
) -> NormalMeanKnownVarResult:
    """Bayesian inference for the normal mean with known variance.

    Model: X_i ~ N(mu, sigma_sq), prior mu ~ N(mu0, sigma_sq / kappa0).

    Parameters
    ----------
    x : array-like
        Sample observations.
    sigma_sq : float
        Known population variance.
    mu0 : float
        Prior mean for mu.
    kappa0 : float
        Prior precision scaling (prior variance = sigma_sq / kappa0).
    alpha : float
        Credible interval level (1 - alpha).

    Returns
    -------
    NormalMeanKnownVarResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x, min_obs=1)
    validate_positive(sigma_sq, "sigma_sq")
    validate_positive(kappa0, "kappa0")
    validate_positive(alpha, "alpha")

    n: int = len(x)
    x_bar: float = float(x.mean())

    kappa_n: float = kappa0 + n
    mu_n: float = (kappa0 * mu0 + n * x_bar) / kappa_n

    posterior_variance: float = sigma_sq / kappa_n
    posterior_std: float = float(np.sqrt(posterior_variance))

    z_alpha2 = norm_ppf(1 - alpha / 2)
    ci_lower: float = mu_n - z_alpha2 * posterior_std
    ci_upper: float = mu_n + z_alpha2 * posterior_std

    predictive_variance: float = sigma_sq * (1 + 1 / kappa_n)
    predictive_std: float = float(np.sqrt(predictive_variance))
    pred_lower: float = mu_n - z_alpha2 * predictive_std
    pred_upper: float = mu_n + z_alpha2 * predictive_std

    return NormalMeanKnownVarResult(
        mu_n=mu_n,
        kappa_n=kappa_n,
        posterior_mean=mu_n,
        posterior_variance=posterior_variance,
        posterior_std=posterior_std,
        credible_interval=(ci_lower, ci_upper),
        predictive_mean=mu_n,
        predictive_variance=predictive_variance,
        predictive_std=predictive_std,
        predictive_interval=(pred_lower, pred_upper),
        n=n,
        x_bar=x_bar,
        alpha=alpha,
    )


def bayes_normal_mean_unknown_var(
    x: np.ndarray,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
    alpha: float = 0.05,
) -> NormalMeanUnknownVarResult:
    """Bayesian inference for the normal mean and variance (Normal-Gamma conjugate).

    Model: X_i ~ N(mu, 1/tau), prior (mu, tau) ~ Normal-Gamma(mu0, kappa0, alpha0, beta0).

    Parameters
    ----------
    x : array-like
        Sample observations.
    mu0 : float
        Prior mean for mu.
    kappa0 : float
        Prior precision scaling.
    alpha0 : float
        Prior shape for Gamma on tau.
    beta0 : float
        Prior rate for Gamma on tau.
    alpha : float
        Credible interval level (1 - alpha).

    Returns
    -------
    NormalMeanUnknownVarResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x, min_obs=1)
    validate_positive(kappa0, "kappa0")
    validate_positive(alpha0, "alpha0")
    validate_positive(beta0, "beta0")
    validate_positive(alpha, "alpha")

    n: int = len(x)
    x_bar: float = float(x.mean())

    kappa_n: float = kappa0 + n
    mu_n: float = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n: float = alpha0 + n / 2.0
    ss: float = float(np.sum((x - x_bar) ** 2))
    beta_n: float = beta0 + 0.5 * ss + (kappa0 * n * (x_bar - mu0) ** 2) / (2.0 * kappa_n)

    posterior_mean_precision: float = alpha_n / beta_n
    posterior_mean_variance: float | None = beta_n / (alpha_n - 1) if alpha_n > 1 else None

    # Marginal posterior for mu: Student-t with df=2*alpha_n, loc=mu_n, scale=sqrt(beta_n/(alpha_n*kappa_n))
    df_mu: float = 2.0 * alpha_n
    scale_mu: float = float(np.sqrt(beta_n / (alpha_n * kappa_n)))
    t_q = t_ppf(1 - alpha / 2, df_mu)
    mu_ci_lower: float = mu_n - t_q * scale_mu
    mu_ci_upper: float = mu_n + t_q * scale_mu

    # Variance CI via Inverse-Gamma: sigma^2 ~ IG(alpha_n, beta_n)
    # Use chi2: 2*beta_n / sigma^2 ~ chi2(2*alpha_n)
    df_var: float = 2.0 * alpha_n
    chi2_lower = chi2_ppf(1 - alpha / 2, df_var)
    chi2_upper = chi2_ppf(alpha / 2, df_var)
    var_ci_lower: float = 2.0 * beta_n / chi2_lower
    var_ci_upper: float = 2.0 * beta_n / chi2_upper

    return NormalMeanUnknownVarResult(
        mu_n=mu_n,
        kappa_n=kappa_n,
        alpha_n=alpha_n,
        beta_n=beta_n,
        posterior_mean_mu=mu_n,
        posterior_mean_precision=posterior_mean_precision,
        posterior_mean_variance=posterior_mean_variance,
        mu_credible_interval=(mu_ci_lower, mu_ci_upper),
        variance_credible_interval=(var_ci_lower, var_ci_upper),
        n=n,
        x_bar=x_bar,
        alpha=alpha,
    )
