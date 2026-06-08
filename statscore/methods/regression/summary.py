"""Regression summary table: coefficient estimates, standard errors, t-stats, p-values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from statscore.methods.regression.least_squares import mult_lr_least_squares, mult_lr_partition_tss
from statscore.utils.distributions import f_pvalue, t_critical, t_pvalue


@dataclass
class RegressionSummaryResult:
    """Full regression summary analogous to R's summary(lm(...))."""

    beta_hat: np.ndarray
    std_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    conf_intervals: list[tuple[float, float]]
    R_squared: float
    adj_R_squared: float
    F_statistic: float
    F_p_value: float
    n: int
    p: int
    df_residual: int
    Se: float
    alpha: float

    def summary(self, feature_names: list[str] | None = None) -> None:
        p = self.p
        if feature_names is None:
            names = ["(Intercept)"] + [f"x{i}" for i in range(1, p)]
        else:
            names = list(feature_names)
            if len(names) < p:
                names = ["(Intercept)"] + names
            if len(names) < p:
                names += [f"x{i}" for i in range(len(names), p)]

        def _stars(pv: float) -> str:
            if pv < 0.001:
                return "***"
            elif pv < 0.01:
                return "**"
            elif pv < 0.05:
                return "*"
            elif pv < 0.1:
                return "."
            return ""

        w = 64
        print("=" * w)
        print("  Regression Summary")
        print("=" * w)
        print(
            f"  Observations: {self.n}    Parameters: {self.p}    df_residual: {self.df_residual}"
        )
        print(
            f"  R² = {self.R_squared:.4f}    "
            f"adj R² = {self.adj_R_squared:.4f}    "
            f"Se = {self.Se:.4f}"
        )
        print("=" * w)
        print(f"  Coefficients (alpha = {self.alpha}):")
        print(f"  {'Variable':<15} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
        print("  " + "-" * (w - 4))
        for i in range(p):
            star = _stars(float(self.p_values[i]))
            print(
                f"  {names[i]:<15} "
                f"{self.beta_hat[i]:>10.4f} "
                f"{self.std_errors[i]:>10.4f} "
                f"{self.t_statistics[i]:>10.4f} "
                f"{self.p_values[i]:>10.4f} {star}"
            )
        print("=" * w)
        df_reg = self.p - 1
        print(
            f"  Overall F({df_reg},{self.df_residual}) = {self.F_statistic:.4f}    "
            f"p = {self.F_p_value:.4f}"
        )
        print("=" * w)
        print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")

    def plot(self, feature_names: list[str] | None = None) -> Figure:
        from matplotlib import pyplot as plt

        p = self.p
        names = feature_names
        if names is None:
            names = ["(Intercept)"] + [f"x{i}" for i in range(1, p)]

        fig, ax = plt.subplots(figsize=(8, max(4, p * 0.6)))
        y_pos = np.arange(p)
        errors_lower = [float(self.beta_hat[i]) - self.conf_intervals[i][0] for i in range(p)]
        errors_upper = [self.conf_intervals[i][1] - float(self.beta_hat[i]) for i in range(p)]

        ax.errorbar(
            [float(b) for b in self.beta_hat], y_pos,
            xerr=[errors_lower, errors_upper],
            fmt="o", color="steelblue", ecolor="steelblue",
            capsize=5, capthick=1.5, markersize=8,
        )
        ax.axvline(0, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Regression Coefficients")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        return fig


def regression_summary(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    feature_names: list[str] | None = None,
) -> RegressionSummaryResult:
    """Compute a full regression summary.

    Parameters
    ----------
    X : array of shape (n, p)
        Design matrix (first column typically all 1s for intercept).
    y : array of shape (n,)
        Response vector.
    alpha : float
        Significance level for confidence intervals.
    feature_names : list[str] or None
        Optional variable names (not stored in result, used only for printing).

    Returns
    -------
    RegressionSummaryResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    ols = mult_lr_least_squares(X, y)
    partition = mult_lr_partition_tss(X, y)

    n, p = X.shape
    df_resid: int = n - p

    Se: float = float(np.sqrt(ols.sigma2_unbiased))
    std_errors: np.ndarray = Se * np.sqrt(np.diag(ols.XtX_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats: np.ndarray = np.where(std_errors > 0, ols.beta_hat / std_errors, 0.0)
    p_values: np.ndarray = np.array([t_pvalue(float(t_stats[i]), df_resid) for i in range(p)])

    t_crit: float = t_critical(alpha / 2, df_resid)
    conf_intervals: list[tuple[float, float]] = [
        (
            float(ols.beta_hat[i] - t_crit * std_errors[i]),
            float(ols.beta_hat[i] + t_crit * std_errors[i]),
        )
        for i in range(p)
    ]

    # Overall F: (RegSS / (p-1)) / (RSS / (n-p))
    df_reg: int = p - 1
    F_stat: float = (
        (partition.RegSS / df_reg) / (partition.RSS / df_resid)
        if (df_reg > 0 and partition.RSS > 0)
        else 0.0
    )
    F_p: float = f_pvalue(F_stat, df_reg, df_resid)

    return RegressionSummaryResult(
        beta_hat=ols.beta_hat,
        std_errors=std_errors,
        t_statistics=t_stats,
        p_values=p_values,
        conf_intervals=conf_intervals,
        R_squared=partition.R_squared,
        adj_R_squared=partition.adj_R_squared,
        F_statistic=F_stat,
        F_p_value=F_p,
        n=n,
        p=p,
        df_residual=df_resid,
        Se=Se,
        alpha=alpha,
    )


