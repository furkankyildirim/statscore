"""Assumption checks and diagnostic tools for statistical inference."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from statscore.regression.least_squares import Mult_LR_Least_squares
from statscore.utils.distributions import norm_ppf, t_critical
from statscore.utils.validation import validate_1d_sample, validate_data_groups


@dataclass
class ShapiroWilkResult:
    """Result of the Shapiro-Wilk normality test."""

    statistic: float
    p_value: float
    reject_H0: bool
    alpha: float
    n: int

    def summary(self) -> None:
        w = 60
        decision_text = (
            "Reject H0 (data not consistent with normality)"
            if self.reject_H0
            else "Fail to reject H0 (data consistent with normality)"
        )
        print("=" * w)
        print("  Shapiro-Wilk Normality Test")
        print("=" * w)
        print(f"  n = {self.n}    alpha = {self.alpha}")
        print("-" * w)
        print(f"  W-statistic: {self.statistic:.4f}    p-value: {self.p_value:.4f}")
        print(f"  Decision:    {decision_text}")
        print("=" * w)


@dataclass
class LeveneResult:
    """Result of Levene's test for homogeneity of variances."""

    statistic: float
    p_value: float
    reject_H0: bool
    alpha: float
    n_groups: int

    def summary(self) -> None:
        w = 60
        decision_text = (
            "Reject H0 (variances differ)"
            if self.reject_H0
            else "Fail to reject H0 (variances are homogeneous)"
        )
        print("=" * w)
        print("  Levene Test for Homogeneity of Variances")
        print("=" * w)
        print(f"  Number of groups: {self.n_groups}    alpha = {self.alpha}")
        print("-" * w)
        print(f"  Levene statistic: {self.statistic:.4f}    p-value: {self.p_value:.4f}")
        print(f"  Decision:         {decision_text}")
        print("=" * w)


@dataclass
class RegressionDiagnosticsResult:
    """Regression diagnostic metrics: leverage, standardized residuals, Cook's D."""

    leverage: np.ndarray
    standardized_residuals: np.ndarray
    cooks_distance: np.ndarray
    high_leverage_mask: np.ndarray
    influential_mask: np.ndarray
    n: int
    p: int

    def summary(self) -> None:
        w = 68
        threshold = 2 * self.p / self.n
        threshold_cook = 4 / self.n
        count_lev = int(self.high_leverage_mask.sum())
        count_infl = int(self.influential_mask.sum())
        print("=" * w)
        print(f"  Regression Diagnostics   (n={self.n}, p={self.p})")
        print("=" * w)
        print(f"  High-leverage points (h_ii > 2p/n = {threshold:.4f}): {count_lev}")
        print(f"  Influential points   (Cook's D > 4/n = {threshold_cook:.4f}): {count_infl}")
        print("-" * w)
        cooks_col = "Cook's D"
        print(
            f"  {'Obs':<5} {'Leverage':>10} {'Std.Resid':>11} {cooks_col:>10} {'Leverage?':>11} {'Influential?':>13}"
        )
        print("-" * w)
        for i in range(self.n):
            lev = float(self.leverage[i])
            sr = float(self.standardized_residuals[i])
            cd = float(self.cooks_distance[i])
            lev_flag = "Yes" if bool(self.high_leverage_mask[i]) else "No"
            infl_flag = "Yes" if bool(self.influential_mask[i]) else "No"
            print(
                f"  {i + 1:<5} {lev:>10.4f} {sr:>11.4f} {cd:>10.4f} {lev_flag:>11} {infl_flag:>13}"
            )
        print("=" * w)


@dataclass
class MeanConfidenceIntervalResult:
    """Confidence interval for the population mean."""

    lower: float
    upper: float
    point_estimate: float
    margin_of_error: float
    alpha: float
    n: int
    method: str

    def summary(self) -> None:
        w = 60
        method_label = "z-interval" if self.method == "z" else "t-interval"
        pct = int((1 - self.alpha) * 100)
        print("=" * w)
        print(f"  Confidence Interval for the Mean  ({method_label})")
        print("=" * w)
        print(f"  n = {self.n}    point estimate = {self.point_estimate:.4f}")
        print(f"  {pct}% CI: ({self.lower:.4f},  {self.upper:.4f})")
        print(f"  Margin of error: {self.margin_of_error:.4f}    alpha = {self.alpha}")
        print("=" * w)


def shapiro_wilk_test(x: np.ndarray, alpha: float = 0.05) -> ShapiroWilkResult:
    """Shapiro-Wilk test for normality.

    H0: The data is normally distributed.

    Parameters
    ----------
    x : array-like
        Sample observations (1-D, 3 <= n <= 5000).
    alpha : float
        Significance level.

    Returns
    -------
    ShapiroWilkResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x, min_obs=3)

    n: int = len(x)
    if n > 5000:
        warnings.warn(
            f"Shapiro-Wilk test is not reliable for n > 5000 (got n={n}).",
            stacklevel=2,
        )

    stat, p_value = stats.shapiro(x)

    return ShapiroWilkResult(
        statistic=float(stat),
        p_value=float(p_value),
        reject_H0=bool(p_value < alpha),
        alpha=alpha,
        n=n,
    )


def levene_test(data: Sequence[np.ndarray], alpha: float = 0.05) -> LeveneResult:
    """Levene's test for equality of variances.

    H0: All group variances are equal.

    Parameters
    ----------
    data : Sequence of array-like
        One array per group.
    alpha : float
        Significance level.

    Returns
    -------
    LeveneResult
    """
    validate_data_groups(data)

    groups: list[np.ndarray] = [np.asarray(g, dtype=float) for g in data]
    stat, p_value = stats.levene(*groups, center="mean")

    return LeveneResult(
        statistic=float(stat),
        p_value=float(p_value),
        reject_H0=bool(p_value < alpha),
        alpha=alpha,
        n_groups=len(groups),
    )


def regression_diagnostics(X: np.ndarray, y: np.ndarray) -> RegressionDiagnosticsResult:
    """Compute regression diagnostic metrics.

    Parameters
    ----------
    X : array of shape (n, p)
        Design matrix.
    y : array of shape (n,)
        Response vector.

    Returns
    -------
    RegressionDiagnosticsResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    ols = Mult_LR_Least_squares(X, y)
    n, p = X.shape

    h: np.ndarray = np.diag(ols.hat_matrix)
    Se: float = float(np.sqrt(ols.sigma2_unbiased))

    denom = Se * np.sqrt(1.0 - h)
    denom = np.where(denom == 0, np.inf, denom)
    std_resid: np.ndarray = ols.residuals / denom

    # Cook's D_i = (e_i^2 / (p * Se^2)) * (h_ii / (1 - h_ii)^2)
    cooks_d: np.ndarray = (ols.residuals**2 / (p * ols.sigma2_unbiased)) * (h / (1.0 - h) ** 2)

    high_leverage_mask: np.ndarray = h > (2.0 * p / n)
    influential_mask: np.ndarray = cooks_d > (4.0 / n)

    return RegressionDiagnosticsResult(
        leverage=h,
        standardized_residuals=std_resid,
        cooks_distance=cooks_d,
        high_leverage_mask=high_leverage_mask,
        influential_mask=influential_mask,
        n=n,
        p=p,
    )


def mean_confidence_interval(
    x: np.ndarray,
    alpha: float = 0.05,
    sigma: float | None = None,
) -> MeanConfidenceIntervalResult:
    """Confidence interval for the population mean.

    Uses z-interval when sigma is provided, t-interval otherwise.

    Parameters
    ----------
    x : array-like
        Sample observations.
    alpha : float
        Significance level (CI level = 1 - alpha).
    sigma : float or None
        Known population standard deviation. If None, uses sample s with t-distribution.

    Returns
    -------
    MeanConfidenceIntervalResult
    """
    x = np.asarray(x, dtype=float)
    validate_1d_sample(x)

    n: int = len(x)
    x_bar: float = float(x.mean())

    if sigma is not None:
        method = "z"
        z_val: float = norm_ppf(1 - alpha / 2)
        margin: float = z_val * (sigma / np.sqrt(n))
    else:
        method = "t"
        df: int = n - 1
        t_val: float = t_critical(alpha / 2, df)
        s: float = float(x.std(ddof=1))
        margin = t_val * (s / np.sqrt(n))

    return MeanConfidenceIntervalResult(
        lower=x_bar - margin,
        upper=x_bar + margin,
        point_estimate=x_bar,
        margin_of_error=margin,
        alpha=alpha,
        n=n,
        method=method,
    )
