"""Result dataclasses for hypothesis tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from statscore.utils.enums import AlternativeHypothesis


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
            ax.fill_between(x, pdf, where=(x <= -crit_abs).tolist(), alpha=0.3, color="crimson", label="Rejection region")
            ax.fill_between(x, pdf, where=(x >= crit_abs).tolist(), alpha=0.3, color="crimson")
            ax.axvline(-crit_abs, color="crimson", linestyle="--", linewidth=1.2)
            ax.axvline(crit_abs, color="crimson", linestyle="--", linewidth=1.2)
        elif self.alternative is AlternativeHypothesis.GREATER:
            ax.fill_between(x, pdf, where=(x >= z_crit).tolist(), alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(z_crit, color="crimson", linestyle="--", linewidth=1.2)
        else:
            ax.fill_between(x, pdf, where=(x <= z_crit).tolist(), alpha=0.3, color="crimson", label="Rejection region")
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
        from statscore.plots import plot_t_test

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
                ax.fill_between(x, pdf, where=(x <= self.chi2_critical_lower).tolist(), alpha=0.3, color="crimson", label="Rejection region")
                ax.axvline(self.chi2_critical_lower, color="crimson", linestyle="--", linewidth=1.2)
            if np.isfinite(self.chi2_critical_upper):
                ax.fill_between(x, pdf, where=(x >= self.chi2_critical_upper).tolist(), alpha=0.3, color="crimson")
                ax.axvline(self.chi2_critical_upper, color="crimson", linestyle="--", linewidth=1.2)
        elif self.alternative is AlternativeHypothesis.GREATER:
            if np.isfinite(self.chi2_critical_upper):
                ax.fill_between(x, pdf, where=(x >= self.chi2_critical_upper).tolist(), alpha=0.3, color="crimson", label="Rejection region")
                ax.axvline(self.chi2_critical_upper, color="crimson", linestyle="--", linewidth=1.2)
        else:
            if self.chi2_critical_lower > 0:
                ax.fill_between(x, pdf, where=(x <= self.chi2_critical_lower).tolist(), alpha=0.3, color="crimson", label="Rejection region")
                ax.axvline(self.chi2_critical_lower, color="crimson", linestyle="--", linewidth=1.2)

        ax.axvline(self.chi2_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"χ² = {self.chi2_statistic:.4f}")
        ax.set_xlabel("χ²")
        ax.set_ylabel("Density")
        ax.set_title("Chi-Squared Variance Test")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


@dataclass
class TTestTwoSampleResult:
    """Result of a two-sample t-test for equality of means."""

    t_statistic: float
    t_critical: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis
    n1: int
    n2: int
    x1_bar: float
    x2_bar: float
    s1: float
    s2: float
    df: int
    equal_var: bool
    pooled_var: float | None

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        variance_label = "pooled" if self.equal_var else "Welch"
        two_sided = self.alternative is AlternativeHypothesis.TWO_SIDED
        crit_str = f"±{self.t_critical:.4f}" if two_sided else f"{self.t_critical:.4f}"
        print("=" * w)
        print(f"  Two-Sample t-Test  ({variance_label})")
        print("=" * w)
        print(f"  n1 = {self.n1}    x̄1 = {self.x1_bar:.4f}    s1 = {self.s1:.4f}")
        print(f"  n2 = {self.n2}    x̄2 = {self.x2_bar:.4f}    s2 = {self.s2:.4f}")
        print(f"  df = {self.df}    Alternative: {self.alternative.value}")
        print("-" * w)
        print(f"  t-statistic: {self.t_statistic:.4f}    t-critical: {crit_str}")
        print(f"  p-value:     {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:    {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_t_test

        variance_label = "pooled" if self.equal_var else "Welch"
        return plot_t_test(
            t_statistic=self.t_statistic,
            t_critical=self.t_critical,
            df=self.df,
            alternative=self.alternative.value,
            title=f"Two-Sample t-Test ({variance_label})",
        )


@dataclass
class TTestPairedResult:
    """Result of a paired t-test."""

    t_statistic: float
    t_critical: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis
    n: int
    d_bar: float
    s_d: float
    df: int

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        two_sided = self.alternative is AlternativeHypothesis.TWO_SIDED
        crit_str = f"±{self.t_critical:.4f}" if two_sided else f"{self.t_critical:.4f}"
        print("=" * w)
        print("  Paired t-Test")
        print("=" * w)
        print(f"  n = {self.n}    d̄ = {self.d_bar:.4f}    s_d = {self.s_d:.4f}    df = {self.df}")
        print(f"  H0: μ_D = 0    Alternative: {self.alternative.value}")
        print("-" * w)
        print(f"  t-statistic: {self.t_statistic:.4f}    t-critical: {crit_str}")
        print(f"  p-value:     {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:    {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_t_test

        return plot_t_test(
            t_statistic=self.t_statistic,
            t_critical=self.t_critical,
            df=self.df,
            alternative=self.alternative.value,
            title="Paired t-Test",
        )


@dataclass
class FTestVariancesResult:
    """Result of an F-test for equality of variances."""

    f_statistic: float
    f_critical_lower: float
    f_critical_upper: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis
    n1: int
    n2: int
    s1_sq: float
    s2_sq: float
    df1: int
    df2: int

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        print("=" * w)
        print("  F-Test for Equality of Variances")
        print("=" * w)
        print(f"  n1 = {self.n1}    s1² = {self.s1_sq:.4f}    df1 = {self.df1}")
        print(f"  n2 = {self.n2}    s2² = {self.s2_sq:.4f}    df2 = {self.df2}")
        print(f"  Alternative: {self.alternative.value}")
        print("-" * w)
        print(f"  F-statistic: {self.f_statistic:.4f}")
        print(
            f"  Critical region: < {self.f_critical_lower:.4f}  or  > {self.f_critical_upper:.4f}"
        )
        print(f"  p-value:     {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:    {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_f_test

        return plot_f_test(
            f_statistic=self.f_statistic,
            f_critical_low=self.f_critical_lower,
            f_critical_up=self.f_critical_upper,
            df1=self.df1,
            df2=self.df2,
            alternative=self.alternative.value,
            title="F-Test for Equality of Variances",
        )
