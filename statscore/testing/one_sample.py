"""One-sample hypothesis tests for normal population parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from statscore.utils.distributions import (
    chi2_critical,
    chi2_pvalue,
    norm_ppf,
    t_cdf,
    t_critical,
    t_pvalue_one_sided,
    z_pvalue,
)
from statscore.utils.enums import AlternativeHypothesis
from statscore.utils.validation import (
    validate_1d_sample,
    validate_alternative,
    validate_positive,
)


@dataclass
class ZTestResult:
    """Result of a one-sample Z-test for the population mean."""

    z_statistic: float
    z_critical: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis
    n: int
    x_bar: float
    mu0: float
    sigma: float

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        two_sided = self.alternative is AlternativeHypothesis.TWO_SIDED
        crit_str = f"±{self.z_critical:.4f}" if two_sided else f"{self.z_critical:.4f}"
        print("=" * w)
        print("  One-Sample Z-Test")
        print("=" * w)
        print(f"  n = {self.n}    x̄ = {self.x_bar:.4f}    σ = {self.sigma:.4f}")
        print(f"  H0: μ = {self.mu0}    Alternative: {self.alternative.value}")
        print("-" * w)
        print(f"  Z-statistic: {self.z_statistic:.4f}    Z-critical: {crit_str}")
        print(f"  p-value:     {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:    {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(-4, 4, 500)
        pdf = stats.norm.pdf(x)
        ax.plot(x, pdf, color="steelblue", linewidth=2)

        z_crit = self.z_critical
        if self.alternative is AlternativeHypothesis.TWO_SIDED:
            crit_abs = abs(z_crit)
            ax.fill_between(x, pdf, where=x <= -crit_abs, alpha=0.3, color="crimson", label="Rejection region")
            ax.fill_between(x, pdf, where=x >= crit_abs, alpha=0.3, color="crimson")
            ax.axvline(-crit_abs, color="crimson", linestyle="--", linewidth=1.2)
            ax.axvline(crit_abs, color="crimson", linestyle="--", linewidth=1.2)
        elif self.alternative is AlternativeHypothesis.GREATER:
            ax.fill_between(x, pdf, where=x >= z_crit, alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(z_crit, color="crimson", linestyle="--", linewidth=1.2)
        else:
            ax.fill_between(x, pdf, where=x <= z_crit, alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(z_crit, color="crimson", linestyle="--", linewidth=1.2)

        ax.axvline(self.z_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"Z = {self.z_statistic:.4f}")
        ax.set_xlabel("Z")
        ax.set_ylabel("Density")
        ax.set_title("One-Sample Z-Test")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


@dataclass
class TTestOneSampleResult:
    """Result of a one-sample t-test for the population mean."""

    t_statistic: float
    t_critical: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis
    n: int
    x_bar: float
    mu0: float
    s: float
    df: int

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        two_sided = self.alternative is AlternativeHypothesis.TWO_SIDED
        crit_str = f"±{self.t_critical:.4f}" if two_sided else f"{self.t_critical:.4f}"
        print("=" * w)
        print("  One-Sample t-Test")
        print("=" * w)
        print(f"  n = {self.n}    x̄ = {self.x_bar:.4f}    s = {self.s:.4f}    df = {self.df}")
        print(f"  H0: μ = {self.mu0}    Alternative: {self.alternative.value}")
        print("-" * w)
        print(f"  t-statistic: {self.t_statistic:.4f}    t-critical: {crit_str}")
        print(f"  p-value:     {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:    {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.utils.plots import plot_t_test

        return plot_t_test(
            t_statistic=self.t_statistic,
            t_critical=self.t_critical,
            df=self.df,
            alternative=self.alternative.value,
            title="One-Sample t-Test",
        )


@dataclass
class Chi2VarianceTestResult:
    """Result of a chi-squared test for the population variance."""

    chi2_statistic: float
    chi2_critical_lower: float
    chi2_critical_upper: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis
    n: int
    s2: float
    sigma0_sq: float
    df: int

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        print("=" * w)
        print("  Chi-Squared Variance Test")
        print("=" * w)
        print(f"  n = {self.n}    s² = {self.s2:.4f}    df = {self.df}")
        print(f"  H0: σ² = {self.sigma0_sq}    Alternative: {self.alternative.value}")
        print("-" * w)
        print(f"  χ²-statistic: {self.chi2_statistic:.4f}")
        print(
            f"  Critical region: < {self.chi2_critical_lower:.4f}  or  > {self.chi2_critical_upper:.4f}"
        )
        print(f"  p-value:      {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:     {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        fig, ax = plt.subplots(figsize=(8, 5))
        x_max = max(self.chi2_statistic * 1.5, stats.chi2.ppf(0.999, self.df))
        x = np.linspace(0.01, x_max, 500)
        pdf = stats.chi2.pdf(x, self.df)
        ax.plot(x, pdf, color="steelblue", linewidth=2, label=f"χ²(df={self.df})")

        if self.alternative is AlternativeHypothesis.TWO_SIDED:
            if self.chi2_critical_lower > 0:
                ax.fill_between(x, pdf, where=x <= self.chi2_critical_lower, alpha=0.3, color="crimson", label="Rejection region")
                ax.axvline(self.chi2_critical_lower, color="crimson", linestyle="--", linewidth=1.2)
            if np.isfinite(self.chi2_critical_upper):
                ax.fill_between(x, pdf, where=x >= self.chi2_critical_upper, alpha=0.3, color="crimson")
                ax.axvline(self.chi2_critical_upper, color="crimson", linestyle="--", linewidth=1.2)
        elif self.alternative is AlternativeHypothesis.GREATER:
            if np.isfinite(self.chi2_critical_upper):
                ax.fill_between(x, pdf, where=x >= self.chi2_critical_upper, alpha=0.3, color="crimson", label="Rejection region")
                ax.axvline(self.chi2_critical_upper, color="crimson", linestyle="--", linewidth=1.2)
        else:
            if self.chi2_critical_lower > 0:
                ax.fill_between(x, pdf, where=x <= self.chi2_critical_lower, alpha=0.3, color="crimson", label="Rejection region")
                ax.axvline(self.chi2_critical_lower, color="crimson", linestyle="--", linewidth=1.2)

        ax.axvline(self.chi2_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"χ² = {self.chi2_statistic:.4f}")
        ax.set_xlabel("χ²")
        ax.set_ylabel("Density")
        ax.set_title("Chi-Squared Variance Test")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def z_test_mean(
    x: np.ndarray,
    mu0: float,
    sigma: float,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> ZTestResult:
    """One-sample Z-test for the mean when population sigma is known.

    Parameters
    ----------
    x : array-like
        Sample observations.
    mu0 : float
        Hypothesized population mean under H0.
    sigma : float
        Known population standard deviation.
    alpha : float
        Significance level.
    alternative : AlternativeHypothesis
        TWO_SIDED, LESS, or GREATER.

    Returns
    -------
    ZTestResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x)
    validate_positive(sigma, "sigma")
    validate_positive(alpha, "alpha")
    validate_alternative(alternative)

    n: int = len(x)
    x_bar: float = float(x.mean())
    z_stat: float = (x_bar - mu0) / (sigma / np.sqrt(n))

    if alternative is AlternativeHypothesis.TWO_SIDED:
        z_crit = norm_ppf(1 - alpha / 2)
        reject = bool(abs(z_stat) > z_crit)
    elif alternative is AlternativeHypothesis.GREATER:
        z_crit = norm_ppf(1 - alpha)
        reject = bool(z_stat > z_crit)
    else:
        z_crit = norm_ppf(alpha)
        reject = bool(z_stat < z_crit)

    p = z_pvalue(z_stat, alternative)

    return ZTestResult(
        z_statistic=z_stat,
        z_critical=z_crit,
        p_value=p,
        reject_H0=reject,
        alpha=alpha,
        alternative=alternative,
        n=n,
        x_bar=x_bar,
        mu0=mu0,
        sigma=sigma,
    )


def t_test_mean(
    x: np.ndarray,
    mu0: float,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> TTestOneSampleResult:
    """One-sample t-test for the mean when population sigma is unknown.

    Parameters
    ----------
    x : array-like
        Sample observations.
    mu0 : float
        Hypothesized population mean under H0.
    alpha : float
        Significance level.
    alternative : AlternativeHypothesis
        TWO_SIDED, LESS, or GREATER.

    Returns
    -------
    TTestOneSampleResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x)
    validate_positive(alpha, "alpha")
    validate_alternative(alternative)

    n: int = len(x)
    df: int = n - 1
    x_bar: float = float(x.mean())
    s: float = float(x.std(ddof=1))
    t_stat: float = (x_bar - mu0) / (s / np.sqrt(n))

    if alternative is AlternativeHypothesis.TWO_SIDED:
        t_crit = t_critical(alpha / 2, df)
        p = float(2.0 * (1.0 - t_cdf(abs(t_stat), df)))
        reject = bool(abs(t_stat) > t_crit)
    elif alternative is AlternativeHypothesis.GREATER:
        t_crit = t_critical(alpha, df)
        p = t_pvalue_one_sided(t_stat, df, alternative)
        reject = bool(t_stat > t_crit)
    else:
        t_crit = -t_critical(alpha, df)
        p = t_pvalue_one_sided(t_stat, df, alternative)
        reject = bool(t_stat < t_crit)

    return TTestOneSampleResult(
        t_statistic=t_stat,
        t_critical=t_crit,
        p_value=p,
        reject_H0=reject,
        alpha=alpha,
        alternative=alternative,
        n=n,
        x_bar=x_bar,
        mu0=mu0,
        s=s,
        df=df,
    )


def chi2_test_variance(
    x: np.ndarray,
    sigma0_sq: float,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> Chi2VarianceTestResult:
    """Chi-squared test for the population variance.

    Parameters
    ----------
    x : array-like
        Sample observations.
    sigma0_sq : float
        Hypothesized population variance under H0.
    alpha : float
        Significance level.
    alternative : AlternativeHypothesis
        TWO_SIDED, LESS, or GREATER.

    Returns
    -------
    Chi2VarianceTestResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x)
    validate_positive(sigma0_sq, "sigma0_sq")
    validate_positive(alpha, "alpha")
    validate_alternative(alternative)

    n: int = len(x)
    df: int = n - 1
    s2: float = float(x.var(ddof=1))
    chi2_stat: float = df * s2 / sigma0_sq

    if alternative is AlternativeHypothesis.TWO_SIDED:
        crit_lower = chi2_critical(1 - alpha / 2, df)
        crit_upper = chi2_critical(alpha / 2, df)
        reject = bool(chi2_stat < crit_lower or chi2_stat > crit_upper)
    elif alternative is AlternativeHypothesis.GREATER:
        crit_lower = 0.0
        crit_upper = chi2_critical(alpha, df)
        reject = bool(chi2_stat > crit_upper)
    else:
        crit_lower = chi2_critical(1 - alpha, df)
        crit_upper = float("inf")
        reject = bool(chi2_stat < crit_lower)

    p = chi2_pvalue(chi2_stat, df, alternative)

    return Chi2VarianceTestResult(
        chi2_statistic=chi2_stat,
        chi2_critical_lower=crit_lower,
        chi2_critical_upper=crit_upper,
        p_value=p,
        reject_H0=reject,
        alpha=alpha,
        alternative=alternative,
        n=n,
        s2=s2,
        sigma0_sq=sigma0_sq,
        df=df,
    )


