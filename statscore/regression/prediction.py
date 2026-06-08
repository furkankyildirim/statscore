"""Prediction confidence intervals for multiple normal linear regression."""

from dataclasses import dataclass

import numpy as np

from statscore.regression.least_squares import LeastSquaresResult, Mult_LR_Least_squares
from statscore.utils.distributions import f_critical, t_critical
from statscore.utils.enums import PredictionMethod
from statscore.utils.validation import validate_design_matrix


@dataclass
class PredictionCIResult:
    """Simultaneous prediction confidence intervals."""

    point_estimates: np.ndarray
    intervals: list[tuple[float, float]]
    half_widths: np.ndarray
    method_used: PredictionMethod


def Mult_norm_LR_pred_CI(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    alpha: float = 0.05,
    method: PredictionMethod = PredictionMethod.BEST,
) -> PredictionCIResult:
    """Simultaneous confidence intervals for d_i^T * beta for rows of D.

    Parameters
    ----------
    X : array of shape (n, k+1)
        Design matrix.
    y : array of shape (n,)
        Response vector.
    D : array of shape (m, k+1)
        Matrix whose rows define prediction points.
    alpha : float
        Significance level.
    method : PredictionMethod
        PredictionMethod.BONFERRONI, .SCHEFFE, or .BEST (narrowest of the two).

    Returns
    -------
    PredictionCIResult with simultaneous CIs holding at 1-alpha.
    """
    if isinstance(method, str):
        method = PredictionMethod(method) if method != "best" else PredictionMethod.BEST

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float)
    validate_design_matrix(X)

    if D.ndim == 1:
        D = D.reshape(1, -1)

    n, p = X.shape
    m: int = D.shape[0]
    df_resid: int = n - p

    if D.shape[1] != p:
        raise ValueError(
            f"D must have {p} columns matching design matrix, got {D.shape[1]}."
        )

    ols: LeastSquaresResult = Mult_LR_Least_squares(X, y)
    Se: float = float(np.sqrt(ols.sigma2_unbiased))

    point_estimates: np.ndarray = D @ ols.beta_hat

    se_pred: np.ndarray = np.array([
        Se * float(np.sqrt(D[i] @ ols.XtX_inv @ D[i])) for i in range(m)
    ])

    scheffe_mult: float = float(np.sqrt(p * f_critical(alpha, p, df_resid)))
    bonf_mult: float = t_critical(alpha / (2 * m), df_resid)

    if method == PredictionMethod.SCHEFFE:
        multiplier: float = scheffe_mult
        method_used: PredictionMethod = PredictionMethod.SCHEFFE
    elif method == PredictionMethod.BONFERRONI:
        multiplier = bonf_mult
        method_used = PredictionMethod.BONFERRONI
    elif method == PredictionMethod.BEST:
        if bonf_mult < scheffe_mult:
            multiplier = bonf_mult
            method_used = PredictionMethod.BONFERRONI
        else:
            multiplier = scheffe_mult
            method_used = PredictionMethod.SCHEFFE
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use PredictionMethod enum."
        )

    half_widths: np.ndarray = multiplier * se_pred
    intervals: list[tuple[float, float]] = [
        (float(point_estimates[i] - half_widths[i]), float(point_estimates[i] + half_widths[i]))
        for i in range(m)
    ]

    return PredictionCIResult(
        point_estimates=point_estimates,
        intervals=intervals,
        half_widths=half_widths,
        method_used=method_used,
    )
