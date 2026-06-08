"""Bayesian conjugate prior inference for normal data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from statscore.utils.distributions import chi2_ppf, norm_ppf, t_ppf
from statscore.utils.validation import validate_1d_sample, validate_positive


@dataclass
class NormalMeanKnownVarResult:
    """Result of Bayesian inference for the normal mean with known variance."""

    mu_n: float
    kappa_n: float
    posterior_mean: float
    posterior_variance: float
    posterior_std: float
    credible_interval: tuple[float, float]
    predictive_mean: float
    predictive_variance: float
    predictive_std: float
    predictive_interval: tuple[float, float]
    n: int
    x_bar: float
    alpha: float

    def summary(self) -> None:
        w = 60
        pct = int((1 - self.alpha) * 100)
        ci_lo, ci_hi = self.credible_interval
        pi_lo, pi_hi = self.predictive_interval
        print("=" * w)
        print("  Bayesian Normal Posterior  (Known Variance)")
        print("=" * w)
        print(f"  n = {self.n}    x̄ = {self.x_bar:.6f}    sigma² known")
        print("-" * (w - 2))
        print("  Posterior hyperparameters:")
        print(f"    mu_n    = {self.mu_n:.6f}")
        print(f"    kappa_n = {self.kappa_n:.6f}")
        print("-" * (w - 2))
        print(f"  Posterior: μ | x ~ N({self.posterior_mean:.6f}, {self.posterior_variance:.6f})")
        print(f"    mean = {self.posterior_mean:.6f}")
        print(f"    std  = {self.posterior_std:.6f}")
        print(f"  {pct}% Credible interval: ({ci_lo:.6f}, {ci_hi:.6f})")
        print("-" * (w - 2))
        print("  Posterior predictive:")
        print(f"    mean = {self.predictive_mean:.6f}")
        print(f"    std  = {self.predictive_std:.6f}")
        print(f"  {pct}% Predictive interval: ({pi_lo:.6f}, {pi_hi:.6f})")
        print("=" * w)


@dataclass
class NormalMeanUnknownVarResult:
    """Result of Bayesian inference for the normal mean with unknown variance."""

    mu_n: float
    kappa_n: float
    alpha_n: float
    beta_n: float
    posterior_mean_mu: float
    posterior_mean_precision: float
    posterior_mean_variance: float | None
    mu_credible_interval: tuple[float, float]
    variance_credible_interval: tuple[float, float]
    n: int
    x_bar: float
    alpha: float

    def summary(self) -> None:
        pct = int((1 - self.alpha) * 100)
        print("=" * 60)
        print("  Bayesian Normal-Gamma Posterior Summary")
        print("=" * 60)
        print(f"  Observations (n):          {self.n}")
        print(f"  Sample mean (x_bar):       {self.x_bar:.6f}")
        print("-" * 60)
        print("  Posterior Hyperparameters:")
        print(f"    mu_n    = {self.mu_n:.6f}")
        print(f"    kappa_n = {self.kappa_n:.6f}")
        print(f"    alpha_n = {self.alpha_n:.6f}")
        print(f"    beta_n  = {self.beta_n:.6f}")
        print("-" * 60)
        print("  Posterior Summaries:")
        print(f"    E[mu]      = {self.posterior_mean_mu:.6f}")
        print(f"    E[tau]     = {self.posterior_mean_precision:.6f}")
        if self.posterior_mean_variance is not None:
            print(f"    E[sigma^2] = {self.posterior_mean_variance:.6f}")
        else:
            print("    E[sigma^2] = undefined (alpha_n <= 1)")
        print("-" * 60)
        print(f"  {pct}% Credible Intervals:")
        print(
            f"    mu:      ({self.mu_credible_interval[0]:.6f}, {self.mu_credible_interval[1]:.6f})"
        )
        print(
            f"    sigma^2: ({self.variance_credible_interval[0]:.6f}, {self.variance_credible_interval[1]:.6f})"
        )
        print("=" * 60)


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
