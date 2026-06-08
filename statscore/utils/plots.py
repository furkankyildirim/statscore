"""Visualization utility functions returning matplotlib Figure objects."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats

_MPL_VER = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])


def plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "Regression Fit",
) -> Figure:
    """Scatter plot with fitted regression line (simple regression)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    beta_hat = np.asarray(beta_hat, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = beta_hat[0] + beta_hat[1] * x_line
    ax.plot(x_line, y_line, color="crimson", linewidth=2, label="Fitted line")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    fitted: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuals vs Fitted",
) -> Figure:
    """Residuals versus fitted values plot."""
    fitted = np.asarray(fitted, dtype=float).ravel()
    residuals = np.asarray(residuals, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, residuals, color="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_qq(
    x: np.ndarray,
    title: str = "Normal Q-Q Plot",
) -> Figure:
    """Normal Q-Q plot."""
    x = np.asarray(x, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
    ax.scatter(osm, osr, color="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)

    line_x = np.array([osm.min(), osm.max()])
    line_y = intercept + slope * line_x
    ax.plot(line_x, line_y, color="crimson", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_anova_groups(
    data: Sequence[np.ndarray],
    group_labels: list[str] | None = None,
    x_label: str = "Group",
    y_label: str = "Value",
    title: str = "Group Distributions",
) -> Figure:
    """Side-by-side box plots with jittered data points for ANOVA groups."""
    groups: list[np.ndarray] = [np.asarray(g, dtype=float) for g in data]
    n_groups = len(groups)

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=(8, 5))
    if _MPL_VER >= (3, 9):
        ax.boxplot(groups, tick_labels=group_labels, widths=0.5)
    else:
        ax.boxplot(groups, labels=group_labels, widths=0.5)  # type: ignore[call-arg]

    for i, g in enumerate(groups):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(g))
        ax.scatter(
            np.full(len(g), i + 1) + jitter,
            g,
            alpha=0.5,
            s=25,
            color="steelblue",
            zorder=3,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_t_test(
    t_statistic: float,
    t_critical: float,
    df: int,
    alternative: str,
    title: str = "t-Test",
) -> Figure:
    """Plot t-test statistic on the t-distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-4.5, 4.5, 500)
    pdf = stats.t.pdf(x, df)
    ax.plot(x, pdf, color="steelblue", linewidth=2, label=f"t(df={df})")

    if alternative == "two-sided":
        crit_abs = abs(t_critical)
        mask_left = x <= -crit_abs
        mask_right = x >= crit_abs
        ax.fill_between(x, pdf, where=mask_left, alpha=0.3, color="crimson", label="Rejection region")
        ax.fill_between(x, pdf, where=mask_right, alpha=0.3, color="crimson")
        ax.axvline(-crit_abs, color="crimson", linestyle="--", linewidth=1.2)
        ax.axvline(crit_abs, color="crimson", linestyle="--", linewidth=1.2)
    elif alternative == "greater":
        mask = x >= t_critical
        ax.fill_between(x, pdf, where=mask, alpha=0.3, color="crimson", label="Rejection region")
        ax.axvline(t_critical, color="crimson", linestyle="--", linewidth=1.2)
    else:
        mask = x <= t_critical
        ax.fill_between(x, pdf, where=mask, alpha=0.3, color="crimson", label="Rejection region")
        ax.axvline(t_critical, color="crimson", linestyle="--", linewidth=1.2)

    ax.axvline(t_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"t = {t_statistic:.4f}")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_f_test(
    f_statistic: float,
    f_critical_low: float,
    f_critical_up: float,
    df1: int,
    df2: int,
    alternative: str,
    title: str = "F-Test",
) -> Figure:
    """Plot F-test statistic on the F-distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_max = max(f_statistic * 1.5, stats.f.ppf(0.999, df1, df2))
    x = np.linspace(0.01, x_max, 500)
    pdf = stats.f.pdf(x, df1, df2)
    ax.plot(x, pdf, color="steelblue", linewidth=2, label=f"F({df1},{df2})")

    if alternative == "two-sided":
        if f_critical_low > 0:
            mask_left = x <= f_critical_low
            ax.fill_between(x, pdf, where=mask_left, alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(f_critical_low, color="crimson", linestyle="--", linewidth=1.2)
        if np.isfinite(f_critical_up):
            mask_right = x >= f_critical_up
            ax.fill_between(x, pdf, where=mask_right, alpha=0.3, color="crimson")
            ax.axvline(f_critical_up, color="crimson", linestyle="--", linewidth=1.2)
    elif alternative == "greater":
        if np.isfinite(f_critical_up):
            mask = x >= f_critical_up
            ax.fill_between(x, pdf, where=mask, alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(f_critical_up, color="crimson", linestyle="--", linewidth=1.2)
    else:
        if f_critical_low > 0:
            mask = x <= f_critical_low
            ax.fill_between(x, pdf, where=mask, alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(f_critical_low, color="crimson", linestyle="--", linewidth=1.2)

    ax.axvline(f_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"F = {f_statistic:.4f}")

    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_simultaneous_ci(
    point_estimates: np.ndarray,
    intervals: list[tuple[float, float]],
    method: str = "",
    labels: list[str] | None = None,
    title: str = "Simultaneous Confidence Intervals",
) -> Figure:
    """Plot multiple simultaneous confidence intervals."""
    m = len(point_estimates)
    if labels is None:
        labels = [f"CI_{i+1}" for i in range(m)]

    fig, ax = plt.subplots(figsize=(8, max(3, m * 0.7)))
    y_pos = np.arange(m)

    for i, (lo, hi) in enumerate(intervals):
        pe = float(point_estimates[i])
        ax.errorbar(
            pe, i,
            xerr=[[pe - lo], [hi - pe]],
            fmt="o", color="steelblue", ecolor="steelblue",
            capsize=5, capthick=1.5, markersize=7,
        )

    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value")
    subtitle = f"Method: {method}" if method else ""
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
