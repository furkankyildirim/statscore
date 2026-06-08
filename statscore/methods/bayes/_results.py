"""Result dataclasses for Bayesian conjugate inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure


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

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        kappa0 = self.kappa_n - self.n
        post_var = self.posterior_variance
        post_std = self.posterior_std
        post_mean = self.mu_n
        prior_var = post_var * self.kappa_n / kappa0 if kappa0 > 0 else post_var * 10
        prior_std = float(np.sqrt(prior_var))
        prior_mean = (post_mean * self.kappa_n - self.n * self.x_bar) / kappa0 if kappa0 > 0 else post_mean

        x_range_sigma = 4.0
        x_min = post_mean - x_range_sigma * max(post_std, prior_std)
        x_max = post_mean + x_range_sigma * max(post_std, prior_std)
        x = np.linspace(x_min, x_max, 500)

        prior_pdf = stats.norm.pdf(x, loc=prior_mean, scale=prior_std)
        post_pdf = stats.norm.pdf(x, loc=post_mean, scale=post_std)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, prior_pdf, color="gray", linestyle="--", linewidth=1.5, label="Prior")
        ax.plot(x, post_pdf, color="steelblue", linewidth=2, label="Posterior")

        ci_lower, ci_upper = self.credible_interval
        ci_mask: list[bool] = ((x >= ci_lower) & (x <= ci_upper)).tolist()
        ax.fill_between(x, post_pdf, where=ci_mask, alpha=0.3, color="steelblue", label="Credible interval")
        ax.axvline(post_mean, color="crimson", linestyle="-", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("μ")
        ax.set_ylabel("Density")
        ax.set_title("Posterior Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


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

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        df_mu = 2.0 * self.alpha_n
        scale_mu = float(np.sqrt(self.beta_n / (self.alpha_n * self.kappa_n)))
        post_mean = self.mu_n
        post_std = scale_mu * float(np.sqrt(df_mu / (df_mu - 2))) if df_mu > 2 else scale_mu * 3

        x_range_sigma = 4.0
        x_min = post_mean - x_range_sigma * post_std
        x_max = post_mean + x_range_sigma * post_std
        x = np.linspace(x_min, x_max, 500)

        post_pdf = stats.t.pdf(x, df=df_mu, loc=post_mean, scale=scale_mu)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, post_pdf, color="steelblue", linewidth=2, label="Posterior (μ)")

        ci_lower, ci_upper = self.mu_credible_interval
        ci_mask: list[bool] = ((x >= ci_lower) & (x <= ci_upper)).tolist()
        ax.fill_between(x, post_pdf, where=ci_mask, alpha=0.3, color="steelblue", label="Credible interval")
        ax.axvline(post_mean, color="crimson", linestyle="-", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("μ")
        ax.set_ylabel("Density")
        ax.set_title("Posterior Distribution (Normal-Gamma)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
