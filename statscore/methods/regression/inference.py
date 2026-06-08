"""Inference for multiple normal linear regression:
simultaneous CIs, confidence regions, and hypothesis tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from statscore.methods.regression._results import (
    ConfidenceRegionResult,
    HypothesisTestResult,
    SimultaneousCIBetaResult,
)
from statscore.methods.regression.least_squares import LeastSquaresResult, mult_lr_least_squares
from statscore.utils.distributions import f_critical, f_pvalue, t_critical
from statscore.utils.enums import PredictionMethod
from statscore.utils.validation import validate_C_matrix, validate_design_matrix


def mult_norm_lr_simul_ci(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> SimultaneousCIBetaResult:
    """Simultaneous confidence intervals for all beta_i's.

    Uses the better of Scheffe's method or Bonferroni correction.

    Parameters
    ----------
    X : array of shape (n, k+1)
    y : array of shape (n,)
    alpha : float

    Returns
    -------
    SimultaneousCIBetaResult with intervals holding simultaneously at 1-alpha.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    validate_design_matrix(X)

    n, p = X.shape
    df_resid: int = n - p

    ols: LeastSquaresResult = mult_lr_least_squares(X, y)
    Se: float = float(np.sqrt(ols.sigma2_unbiased))
    diag_XtX_inv: np.ndarray = np.diag(ols.XtX_inv)
    se_beta: np.ndarray = Se * np.sqrt(diag_XtX_inv)

    scheffe_mult: float = float(np.sqrt(p * f_critical(alpha, p, df_resid)))
    bonf_mult: float = t_critical(alpha / (2 * p), df_resid)

    if bonf_mult < scheffe_mult:
        multiplier: float = bonf_mult
        method_used: PredictionMethod = PredictionMethod.BONFERRONI
    else:
        multiplier = scheffe_mult
        method_used = PredictionMethod.SCHEFFE

    half_widths: np.ndarray = multiplier * se_beta
    intervals: list[tuple[float, float]] = [
        (float(ols.beta_hat[i] - half_widths[i]), float(ols.beta_hat[i] + half_widths[i]))
        for i in range(p)
    ]

    return SimultaneousCIBetaResult(
        beta_hat=ols.beta_hat,
        intervals=intervals,
        half_widths=half_widths,
        method=method_used,
    )


def mult_norm_lr_cr(
    X: np.ndarray, y: np.ndarray, C: np.ndarray, alpha: float = 0.05
) -> ConfidenceRegionResult:
    """Compute the confidence region (ellipsoid) for C*beta.

    CR = {C*beta : (C*beta_hat - C*beta)^T [C(X^TX)^{-1}C^T]^{-1} (C*beta_hat - C*beta)
          <= r * Se^2 * f_{alpha,r,n-k-1}}

    Parameters
    ----------
    X : array of shape (n, k+1)
    y : array of shape (n,)
    C : array of shape (r, k+1) with rank r
    alpha : float

    Returns
    -------
    ConfidenceRegionResult specifying the ellipsoid parameters.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    C = np.asarray(C, dtype=float)
    validate_design_matrix(X)
    validate_C_matrix(C, X)

    n, p = X.shape
    r: int = C.shape[0]
    df_resid: int = n - p

    ols: LeastSquaresResult = mult_lr_least_squares(X, y)
    center: np.ndarray = C @ ols.beta_hat
    shape_matrix: np.ndarray = C @ ols.XtX_inv @ C.T
    F_crit: float = f_critical(alpha, r, df_resid)
    radius_sq: float = r * ols.sigma2_unbiased * F_crit

    return ConfidenceRegionResult(
        center=center,
        shape_matrix=shape_matrix,
        radius_squared=radius_sq,
        Se2=ols.sigma2_unbiased,
        F_critical=F_crit,
        r=r,
        df_residual=df_resid,
    )


def mult_norm_lr_is_in_cr(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    c0: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """Test whether c0 is inside the confidence region for C*beta.

    Parameters
    ----------
    X : array of shape (n, k+1)
    y : array of shape (n,)
    C : array of shape (r, k+1)
    c0 : array of shape (r,)
    alpha : float

    Returns
    -------
    True if c0 is in the 100(1-alpha)% confidence region.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    C = np.asarray(C, dtype=float)
    c0 = np.asarray(c0, dtype=float).ravel()
    validate_design_matrix(X)
    validate_C_matrix(C, X)

    n, p = X.shape
    r: int = C.shape[0]
    df_resid: int = n - p

    ols: LeastSquaresResult = mult_lr_least_squares(X, y)
    diff: np.ndarray = C @ ols.beta_hat - c0
    shape_inv: np.ndarray = np.linalg.inv(C @ ols.XtX_inv @ C.T)
    quadratic_form: float = float(diff @ shape_inv @ diff)

    F_crit: float = f_critical(alpha, r, df_resid)
    threshold: float = r * ols.sigma2_unbiased * F_crit

    return bool(quadratic_form <= threshold)


def mult_norm_lr_test_general(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    c0: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Test H0: C*beta = c0 vs H1: C*beta != c0.

    Uses the F-test:
    F = [(C*beta_hat - c0)^T [C(X^TX)^{-1}C^T]^{-1} (C*beta_hat - c0)] / (r * Se^2)

    Parameters
    ----------
    X : array of shape (n, k+1)
    y : array of shape (n,)
    C : array of shape (r, k+1) with rank r
    c0 : array of shape (r,)
    alpha : float

    Returns
    -------
    HypothesisTestResult with F-statistic, critical value, p-value, decision.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    C = np.asarray(C, dtype=float)
    c0 = np.asarray(c0, dtype=float).ravel()
    validate_design_matrix(X)
    validate_C_matrix(C, X)

    n, p = X.shape
    r: int = C.shape[0]
    df_resid: int = n - p

    if len(c0) != r:
        raise ValueError(f"c0 must have {r} elements, got {len(c0)}.")

    ols: LeastSquaresResult = mult_lr_least_squares(X, y)
    diff: np.ndarray = C @ ols.beta_hat - c0
    shape_inv: np.ndarray = np.linalg.inv(C @ ols.XtX_inv @ C.T)
    quadratic_form: float = float(diff @ shape_inv @ diff)

    F_stat: float = quadratic_form / (r * ols.sigma2_unbiased)
    F_crit: float = f_critical(alpha, r, df_resid)
    p_val: float = f_pvalue(F_stat, r, df_resid)

    return HypothesisTestResult(
        test_statistic=F_stat,
        F_critical=F_crit,
        p_value=p_val,
        reject_H0=(F_stat > F_crit),
        alpha=alpha,
        df_numerator=r,
        df_denominator=df_resid,
    )


def mult_norm_lr_test_comp(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    components: Sequence[int],
) -> HypothesisTestResult:
    """Test H0: beta_{j1} = ... = beta_{jr} = 0 vs H1: not H0.

    Parameters
    ----------
    X : array of shape (n, k+1)
    y : array of shape (n,)
    alpha : float
    components : Sequence[int]
        Indices j1,...,jr in {0,...,k} of the components to test.

    Returns
    -------
    HypothesisTestResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    validate_design_matrix(X)

    _, p = X.shape
    components_list: list[int] = list(components)
    r: int = len(components_list)

    for j in components_list:
        if j < 0 or j >= p:
            raise ValueError(f"Component index {j} out of range [0, {p - 1}].")

    C: np.ndarray = np.zeros((r, p))
    for i, j in enumerate(components_list):
        C[i, j] = 1.0

    c0: np.ndarray = np.zeros(r)
    return mult_norm_lr_test_general(X, y, C, c0, alpha)


def mult_norm_lr_test_linear_reg(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> HypothesisTestResult:
    """Test existence of linear regression: H0: beta_1=...=beta_k=0.

    Parameters
    ----------
    X : array of shape (n, k+1)
        Design matrix (first column = intercept).
    y : array of shape (n,)
    alpha : float

    Returns
    -------
    HypothesisTestResult
    """
    X = np.asarray(X, dtype=float)
    _, p = X.shape

    components: list[int] = list(range(1, p))
    return mult_norm_lr_test_comp(X, y, alpha, components)
