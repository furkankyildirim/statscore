"""Result dataclasses for diagnostic procedures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure


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

    def plot(self, x: np.ndarray) -> Figure:
        from statscore.plots import plot_qq

        return plot_qq(x, title="Normal Q-Q Plot (Shapiro-Wilk)")


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

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        threshold_lev = 2 * self.p / self.n
        threshold_cook = 4 / self.n
        obs = np.arange(1, self.n + 1)

        ax1 = axes[0, 0]
        ax1.bar(obs, self.leverage, color="steelblue", edgecolor="k", linewidth=0.3)
        ax1.axhline(threshold_lev, color="crimson", linestyle="--", linewidth=1.2, label=f"2p/n = {threshold_lev:.4f}")
        ax1.set_xlabel("Observation")
        ax1.set_ylabel("Leverage")
        ax1.set_title("Leverage (h_ii)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.scatter(obs, self.standardized_residuals, color="steelblue", edgecolors="k", linewidths=0.5)
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8)
        ax2.axhline(2, color="crimson", linestyle="--", linewidth=1)
        ax2.axhline(-2, color="crimson", linestyle="--", linewidth=1)
        ax2.set_xlabel("Observation")
        ax2.set_ylabel("Std. Residual")
        ax2.set_title("Standardized Residuals")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.bar(obs, self.cooks_distance, color="coral", edgecolor="k", linewidth=0.3)
        ax3.axhline(threshold_cook, color="crimson", linestyle="--", linewidth=1.2, label=f"4/n = {threshold_cook:.4f}")
        ax3.set_xlabel("Observation")
        ax3.set_ylabel("Cook's D")
        ax3.set_title("Cook's Distance")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.scatter(self.leverage, self.standardized_residuals, color="steelblue", edgecolors="k", linewidths=0.5)
        ax4.axhline(0, color="gray", linestyle="-", linewidth=0.8)
        ax4.axvline(threshold_lev, color="crimson", linestyle="--", linewidth=1)
        ax4.set_xlabel("Leverage")
        ax4.set_ylabel("Std. Residual")
        ax4.set_title("Residuals vs Leverage")
        ax4.grid(True, alpha=0.3)

        fig.suptitle("Regression Diagnostics", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


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

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt

        method_label = "z-interval" if self.method == "z" else "t-interval"
        pct = int((1 - self.alpha) * 100)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.errorbar(
            self.point_estimate, 0,
            xerr=[[self.point_estimate - self.lower], [self.upper - self.point_estimate]],
            fmt="o", color="steelblue", ecolor="steelblue",
            capsize=8, capthick=2, markersize=10, linewidth=2,
        )
        ax.axvline(self.point_estimate, color="steelblue", linestyle="-", linewidth=0.8, alpha=0.5)
        label = f"{method_label} — {pct}% CI: ({self.lower:.4f}, {self.upper:.4f})"
        ax.set_title(f"Confidence Interval for the Mean\n{label}")
        ax.set_yticks([])
        ax.set_xlabel("Value")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_ylim(-1, 1)
        fig.tight_layout()
        return fig


__all__ = [
    "ShapiroWilkResult",
    "LeveneResult",
    "RegressionDiagnosticsResult",
    "MeanConfidenceIntervalResult",
]
