"""Two-sample hypothesis tests for normal population parameters."""

from dataclasses import dataclass

import numpy as np

from statscore.utils.distributions import (
    f_critical,
    f_critical_lower,
    f_pvalue,
    f_pvalue_lower,
    t_cdf,
    t_critical,
    t_pvalue_one_sided,
)
from statscore.utils.enums import AlternativeHypothesis
from statscore.utils.validation import (
    validate_1d_sample,
    validate_alternative,
    validate_positive,
)


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
        print(f"  Critical region: < {self.f_critical_lower:.4f}  or  > {self.f_critical_upper:.4f}")
        print(f"  p-value:     {self.p_value:.4f}    alpha: {self.alpha}")
        print(f"  Decision:    {decision}")
        print("=" * w)


def t_test_two_sample(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    equal_var: bool = True,
) -> TTestTwoSampleResult:
    """Two-sample t-test for equality of means.

    Parameters
    ----------
    x1, x2 : array-like
        Sample observations from two populations.
    alpha : float
        Significance level.
    alternative : AlternativeHypothesis
        TWO_SIDED, LESS, or GREATER.
    equal_var : bool
        If True, assume equal variances (pooled). If False, use Welch's t-test.

    Returns
    -------
    TTestTwoSampleResult
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    validate_1d_sample(x1, "x1")
    validate_1d_sample(x2, "x2")
    validate_positive(alpha, "alpha")
    validate_alternative(alternative)

    n1: int = len(x1)
    n2: int = len(x2)
    x1_bar: float = float(x1.mean())
    x2_bar: float = float(x2.mean())
    s1: float = float(x1.std(ddof=1))
    s2: float = float(x2.std(ddof=1))

    if equal_var:
        sp2: float = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        df: int = n1 + n2 - 2
        se: float = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        pooled_var: float | None = sp2
    else:
        se_sq: float = s1**2 / n1 + s2**2 / n2
        se = np.sqrt(se_sq)
        nu: float = se_sq**2 / (
            (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        )
        df = int(np.floor(nu))
        pooled_var = None

    t_stat: float = (x1_bar - x2_bar) / se

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

    return TTestTwoSampleResult(
        t_statistic=t_stat,
        t_critical=t_crit,
        p_value=p,
        reject_H0=reject,
        alpha=alpha,
        alternative=alternative,
        n1=n1,
        n2=n2,
        x1_bar=x1_bar,
        x2_bar=x2_bar,
        s1=s1,
        s2=s2,
        df=df,
        equal_var=equal_var,
        pooled_var=pooled_var,
    )


def t_test_paired(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> TTestPairedResult:
    """Paired t-test for equality of means (H0: mu_D = 0).

    Parameters
    ----------
    x1, x2 : array-like
        Paired sample observations.
    alpha : float
        Significance level.
    alternative : AlternativeHypothesis
        TWO_SIDED, LESS, or GREATER.

    Returns
    -------
    TTestPairedResult
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    validate_positive(alpha, "alpha")
    validate_alternative(alternative)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("x1 and x2 must be 1-D arrays.")
    if len(x1) != len(x2):
        raise ValueError(
            f"x1 and x2 must have the same length, got {len(x1)} and {len(x2)}."
        )
    if len(x1) < 2:
        raise ValueError("Samples must have at least 2 observations.")

    d: np.ndarray = x1 - x2
    n: int = len(d)
    df: int = n - 1
    d_bar: float = float(d.mean())
    s_d: float = float(d.std(ddof=1))
    t_stat: float = d_bar / (s_d / np.sqrt(n))

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

    return TTestPairedResult(
        t_statistic=t_stat,
        t_critical=t_crit,
        p_value=p,
        reject_H0=reject,
        alpha=alpha,
        alternative=alternative,
        n=n,
        d_bar=d_bar,
        s_d=s_d,
        df=df,
    )


def f_test_variances(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> FTestVariancesResult:
    """F-test for equality of two population variances.

    Parameters
    ----------
    x1, x2 : array-like
        Sample observations from two populations.
    alpha : float
        Significance level.
    alternative : AlternativeHypothesis
        TWO_SIDED, LESS, or GREATER.

    Returns
    -------
    FTestVariancesResult
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    validate_1d_sample(x1, "x1")
    validate_1d_sample(x2, "x2")
    validate_positive(alpha, "alpha")
    validate_alternative(alternative)

    n1: int = len(x1)
    n2: int = len(x2)
    s1_sq: float = float(x1.var(ddof=1))
    s2_sq: float = float(x2.var(ddof=1))
    df1: int = n1 - 1
    df2: int = n2 - 1
    f_stat: float = s1_sq / s2_sq

    if alternative is AlternativeHypothesis.TWO_SIDED:
        crit_lower = f_critical_lower(alpha / 2, df1, df2)
        crit_upper = f_critical(alpha / 2, df1, df2)
        p_lower = f_pvalue_lower(f_stat, df1, df2)
        p_upper = f_pvalue(f_stat, df1, df2)
        p = float(2.0 * min(p_lower, p_upper))
        reject = bool(f_stat < crit_lower or f_stat > crit_upper)
    elif alternative is AlternativeHypothesis.GREATER:
        crit_lower = 0.0
        crit_upper = f_critical(alpha, df1, df2)
        p = f_pvalue(f_stat, df1, df2)
        reject = bool(f_stat > crit_upper)
    else:
        crit_lower = f_critical_lower(alpha, df1, df2)
        crit_upper = float("inf")
        p = f_pvalue_lower(f_stat, df1, df2)
        reject = bool(f_stat < crit_lower)

    return FTestVariancesResult(
        f_statistic=f_stat,
        f_critical_lower=crit_lower,
        f_critical_upper=crit_upper,
        p_value=p,
        reject_H0=reject,
        alpha=alpha,
        alternative=alternative,
        n1=n1,
        n2=n2,
        s1_sq=s1_sq,
        s2_sq=s2_sq,
        df1=df1,
        df2=df2,
    )
