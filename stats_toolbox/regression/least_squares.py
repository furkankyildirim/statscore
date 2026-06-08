"""Multiple linear regression: OLS estimation and TSS partitioning."""

from dataclasses import dataclass

import numpy as np

from stats_toolbox.utils.validation import validate_design_matrix


@dataclass
class LeastSquaresResult:
    """Result of OLS estimation for multiple linear regression."""

    beta_hat: np.ndarray
    sigma2_mle: float
    sigma2_unbiased: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    hat_matrix: np.ndarray
    XtX_inv: np.ndarray


@dataclass
class PartitionTSSResult:
    """Partition of total sum of squares in regression."""

    TSS: float
    RegSS: float
    RSS: float
    R_squared: float


def Mult_LR_Least_squares(X: np.ndarray, y: np.ndarray) -> LeastSquaresResult:
    """Find the least squares solution for multiple linear regression.

    Y = X*beta + e

    Parameters
    ----------
    X : array-like of shape (n, k+1)
        Design matrix (typically first column is all 1s for intercept).
    y : array-like of shape (n,) or (n,1)
        Response vector.

    Returns
    -------
    LeastSquaresResult with beta_hat (MLE), sigma2_mle, sigma2_unbiased (Se^2),
    residuals, fitted values, hat matrix, and (X^T X)^{-1}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    validate_design_matrix(X)

    n, p = X.shape

    if len(y) != n:
        raise ValueError(f"y must have {n} elements, got {len(y)}.")

    XtX: np.ndarray = X.T @ X
    XtX_inv: np.ndarray = np.linalg.inv(XtX)
    beta_hat: np.ndarray = XtX_inv @ (X.T @ y)

    fitted: np.ndarray = X @ beta_hat
    residuals: np.ndarray = y - fitted

    RSS: float = float(residuals @ residuals)
    sigma2_mle: float = RSS / n
    sigma2_unbiased: float = RSS / (n - p)

    H: np.ndarray = X @ XtX_inv @ X.T

    return LeastSquaresResult(
        beta_hat=beta_hat,
        sigma2_mle=sigma2_mle,
        sigma2_unbiased=sigma2_unbiased,
        residuals=residuals,
        fitted_values=fitted,
        hat_matrix=H,
        XtX_inv=XtX_inv,
    )


def Mult_LR_partition_TSS(X: np.ndarray, y: np.ndarray) -> PartitionTSSResult:
    """Partition total sum of squares into regression SS and residual SS.

    TSS = RegSS + RSS

    Parameters
    ----------
    X : array-like of shape (n, k+1)
        Design matrix.
    y : array-like of shape (n,)
        Response vector.

    Returns
    -------
    PartitionTSSResult with TSS, RegSS, RSS, R_squared.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    result: LeastSquaresResult = Mult_LR_Least_squares(X, y)

    y_bar: float = float(y.mean())
    TSS: float = float(np.sum((y - y_bar) ** 2))
    RSS: float = float(np.sum(result.residuals ** 2))
    RegSS: float = TSS - RSS
    R_squared: float = RegSS / TSS if TSS > 0 else 0.0

    return PartitionTSSResult(TSS=TSS, RegSS=RegSS, RSS=RSS, R_squared=R_squared)
