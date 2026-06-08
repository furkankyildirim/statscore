"""Assumption checks and diagnostic tools for statistical inference."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import stats

from statscore.methods.diagnostics._results import (
    LeveneResult,
    MeanConfidenceIntervalResult,
    RegressionDiagnosticsResult,
    ShapiroWilkResult,
)
from statscore.methods.regression import mult_lr_least_squares
from statscore.utils.distributions import norm_ppf, t_critical
from statscore.utils.validation import validate_1d_sample, validate_data_groups


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

    ols = mult_lr_least_squares(X, y)
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


__all__ = [
    "ShapiroWilkResult",
    "LeveneResult",
    "RegressionDiagnosticsResult",
    "MeanConfidenceIntervalResult",
    "shapiro_wilk_test",
    "levene_test",
    "regression_diagnostics",
    "mean_confidence_interval",
]
